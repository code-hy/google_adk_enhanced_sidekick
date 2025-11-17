from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional
from sidekick_adk import SidekickADK
from observability import setup_tracing, logger, request_counter
from a2a_protocol import A2AServer, A2AMessage
from pathlib import Path
import os

active_sidekicks: Dict[str, SidekickADK] = {}
a2a_server = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Enhanced Sidekick ADK Server")
    setup_tracing("sidekick-adk")
    
    global a2a_server
    if active_sidekicks:
        a2a_server = A2AServer(next(iter(active_sidekicks.values())).agent)
    
    yield
    
    # Shutdown
    logger.info("Shutting down and cleaning up resources")
    for sid, sidekick in active_sidekicks.items():
        await sidekick.cleanup()

app = FastAPI(title="Enhanced Sidekick ADK API", lifespan=lifespan)

# Serve static files
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

html = """... (enhanced HTML with metrics dashboard) ..."""

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.utcnow()
    response = await call_next(request)
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    request_counter.add(1, {
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code
    })
    
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_seconds=duration
    )
    
    return response

@app.get("/")
async def get_chat_interface():
    return HTMLResponse(html)

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus-style metrics"""
    return JSONResponse({
        "active_sessions": len(active_sidekicks),
        "memory_usage": len(active_sidekicks) * 100,  # Mock metric
        "total_requests": 1  # Would be tracked properly
    })

@app.post("/pause/{session_id}")
async def pause_agent(session_id: str):
    """Pause a running agent"""
    if session_id not in active_sidekicks:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sidekick = active_sidekicks[session_id]
    await sidekick.pause()
    
    return {"message": "Agent paused", "session_id": session_id}

@app.post("/resume/{session_id}")
async def resume_agent(session_id: str):
    """Resume a paused agent"""
    if session_id not in active_sidekicks:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sidekick = active_sidekicks[session_id]
    sidekick.resume()
    
    return {"message": "Agent resumed", "session_id": session_id}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: Optional[str] = Query(None)):
    await websocket.accept()
    
    # Initialize A2A server if needed
    global a2a_server
    if not a2a_server and active_sidekicks:
        a2a_server = A2AServer(next(iter(active_sidekicks.values())).agent)
    
    if session_id and session_id in active_sidekicks:
        sidekick = active_sidekicks[session_id]
        await websocket.send_json({
            "type": "reconnected",
            "session_id": session_id,
            "content": "Reconnected to existing session"
        })
    else:
        sidekick = SidekickADK()
        await sidekick.setup()
        session_id = sidekick.session_id
        active_sidekicks[session_id] = sidekick
        a2a_server = A2AServer(sidekick.agent)
        
        await websocket.send_json({
            "type": "initialized",
            "session_id": session_id,
            "content": "New session initialized"
        })
    
    current_history = []
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("action") == "process":
                user_message = message_data["message"]
                success_criteria = message_data.get("success_criteria", "")
                
                await websocket.send_json({
                    "role": "user",
                    "content": user_message,
                    "type": "user_message"
                })
                
                result = await sidekick.run_superstep(
                    message=user_message,
                    success_criteria=success_criteria,
                    history=current_history
                )
                
                await websocket.send_json({
                    "role": "assistant",
                    "content": result["worker_response"],
                    "type": "worker_response"
                })
                
                await websocket.send_json({
                    "role": "evaluator",
                    "content": f"Score: {result['quality_score']:.2f} - {result['evaluator_feedback']}",
                    "type": "evaluator_feedback"
                })
                
                current_history.extend(result["conversation_update"])
                
                await websocket.send_json({
                    "role": "system",
                    "content": {
                        "success_criteria_met": result["success_criteria_met"],
                        "user_input_needed": result["user_input_needed"],
                        "quality_score": result["quality_score"]
                    },
                    "type": "status_update"
                })
                
            elif message_data.get("action") == "reset":
                await sidekick.reset()
                new_session_id = sidekick.session_id
                
                del active_sidekicks[session_id]
                active_sidekicks[new_session_id] = sidekick
                session_id = new_session_id
                a2a_server = A2AServer(sidekick.agent)
                
                current_history = []
                
                await websocket.send_json({
                    "type": "reset_complete",
                    "session_id": new_session_id,
                    "content": "Session reset complete"
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"Websocket error: {e}", session_id=session_id)
        await websocket.send_json({
            "type": "error",
            "content": f"Error: {str(e)}"
        })

@app.post("/a2a")
async def a2a_protocol_endpoint(message: Dict[str, Any]):
    """A2A Protocol endpoint"""
    if not a2a_server:
        raise HTTPException(status_code=503, detail="A2A not initialized")
    
    return await a2a_server.handle_message(message)

@app.get("/agent-card")
async def get_agent_card():
    """Get A2A agent card"""
    if not a2a_server:
        raise HTTPException(status_code=503, detail="A2A not initialized")
    
    return a2a_server.generate_agent_card()

@app.get("/session/{session_id}/memory")
async def get_session_memory(session_id: str):
    """Get session memory summary"""
    if session_id not in active_sidekicks:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sidekick = active_sidekicks[session_id]
    summary = sidekick.memory_bank.get_session_summary(session_id)
    
    return {"session_id": session_id, "memory_summary": summary}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(active_sidekicks),
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
