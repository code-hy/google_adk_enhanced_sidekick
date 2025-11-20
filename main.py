# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from sidekick_adk import SidekickADK
from observability import setup_tracing, logger, request_counter
from a2a_protocol import A2AServer, A2AMessage
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()


active_sidekicks: Dict[str, SidekickADK] = {}
a2a_server = None
# Create required directories on startup
REQUIRED_DIRS = ["static", "sandbox", "memory_bank", "agent_states"]
for dir_name in REQUIRED_DIRS:
    Path(dir_name).mkdir(exist_ok=True)
    print(f"✅ Ensured directory exists: {dir_name}")


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

# Add a handler for /v1/models to prevent 404 errors
@app.get("/v1/models")
async def get_models():
    """Return a mock models response to prevent 404 errors"""
    return {
        "object": "list",
        "data": [
            {
                "id": "gemini-1.5-flash",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "google"
            },
            {
                "id": "gemini-1.5-pro",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "google"
            }
        ]
    }

html = """... (enhanced HTML with metrics dashboard) ..."""
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Sidekick ADK Chat</title>
    <style>
        body { font-family: system-ui; max-width: 900px; margin: 40px auto; padding: 20px; background: #f8f9fa; }
        #messages { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; background: white; border-radius: 8px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .user { background: #e3f2fd; text-align: right; margin-left: 20%; }
        .assistant { background: #f3e5f5; margin-right: 20%; }
        .evaluator { background: #fff3e0; font-style: italic; margin-right: 15%; margin-left: 15%; }
        input, textarea { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 4px; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; border: none; border-radius: 4px; }
        .primary { background: #4CAF50; color: white; }
        .stop { background: #f44336; color: white; }
        .controls { display: flex; gap: 10px; margin-top: 10px; }
        #status { font-size: 0.8em; color: #666; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🤖 Sidekick ADK - Personal Co-Worker</h1>
    <p>Your AI assistant for research, coding, and task automation.</p>
    
    <div id="messages"></div>
    
    <div>
        <input type="text" id="sessionId" placeholder="Session ID (leave empty for new session)" readonly />
        <textarea id="successCriteria" placeholder="What are your success criteria? (e.g., 'Save a summary to file')"></textarea>
        <textarea id="message" placeholder="Your request to the Sidekick..." rows="3"></textarea>
    </div>
    
    <div class="controls">
        <button id="connectBtn" class="primary">Connect</button>
        <button id="goBtn" class="primary">🚀 Submit</button>
        <button id="resetBtn" class="stop">Reset</button>
        <button id="clearBtn" class="stop">Clear Chat</button>
    </div>
    
    <div id="status">Status: Not connected</div>

    <script>
        let ws = null;
        let sessionId = null;
        
        document.getElementById('connectBtn').onclick = () => {
            const inputSessionId = document.getElementById('sessionId').value;
            const url = `ws://localhost:8000/ws${inputSessionId ? '?session_id=' + inputSessionId : ''}`;
            
            ws = new WebSocket(url);
            
            ws.onopen = () => {
                document.getElementById('status').innerHTML = '✅ Connected';
                ws.send(JSON.stringify({action: 'initialize'}));
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'initialized') {
                    sessionId = data.session_id;
                    document.getElementById('sessionId').value = sessionId;
                }
                
                addMessage(data.role || 'assistant', data.content, data.type);
            };
            
            ws.onclose = () => {
                document.getElementById('status').innerHTML = '❌ Disconnected';
            };
            
            ws.onerror = (error) => {
                document.getElementById('status').innerHTML = '❌ Connection error';
                console.error('WebSocket error:', error);
            };
        };
        
        document.getElementById('goBtn').onclick = () => {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                alert('Not connected. Click Connect first.');
                return;
            }
            
            const message = document.getElementById('message').value;
            const successCriteria = document.getElementById('successCriteria').value;
            
            ws.send(JSON.stringify({
                action: 'process',
                message: message,
                success_criteria: successCriteria
            }));
            
            addMessage('user', message, 'user_message');
            document.getElementById('message').value = '';
            document.getElementById('status').innerHTML = '🤔 Processing...';
        };
        
        document.getElementById('resetBtn').onclick = () => {
            if (!ws) return;
            ws.send(JSON.stringify({action: 'reset'}));
            document.getElementById('messages').innerHTML = '';
            document.getElementById('status').innerHTML = '🔄 Session reset';
        };
        
        document.getElementById('clearBtn').onclick = () => {
            document.getElementById('messages').innerHTML = '';
        };
        
        function addMessage(role, content, type = '') {
            const messages = document.getElementById('messages');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role} ${type || ''}`;
            
            const time = new Date().toLocaleTimeString();
            msgDiv.innerHTML = `<strong>${role.toUpperCase()} [${time}]</strong><br/>${content}`;
            
            messages.appendChild(msgDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        // Auto-connect on load
        window.onload = () => {
            document.getElementById('connectBtn').click();
        };
    </script>
</body>
</html>
"""

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.utcnow()
    response = await call_next(request)
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    try:
        request_counter.add(1, {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code
        })
    except:
        pass  # Ignore metric recording errors
    
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_seconds=duration
    )
    
    return response

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the main chat interface"""
    return HTMLResponse(content=html, status_code=200)

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

# In main.py, find the websocket_endpoint function and update the message handling:

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
                
                # Don't add user message here - let the frontend handle it
                # await websocket.send_json({
                #     "role": "user",
                #     "content": user_message,
                #     "type": "user_message"
                # })
                
                try:
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
                except Exception as e:
                    await websocket.send_json({
                        "role": "assistant",
                        "content": f"Error processing request: {str(e)}",
                        "type": "error"
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
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Error: {str(e)}"
            })
        except:
            pass

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



if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Sidekick ADK Server...")
    print("📱 Access the UI at: http://localhost:8000")
    print("📡 Health check: http://localhost:8000/health")
    
    # uvicorn.run(
    #     "main:app",  # Module:app_name
    #     host="0.0.0.0",
    #     port=8000,
    #     reload=True,  # Auto-reload on code changes
    #     log_level="info"
    # )
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],  # Watch the current directory for our source code
        reload_excludes=["./sandbox/*", "./agent_states/*", "./memory_bank/*"], # But ignore these
        log_level="info"
    )