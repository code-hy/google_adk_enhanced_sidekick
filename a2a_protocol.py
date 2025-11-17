from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import json
import uuid
import hashlib
import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class A2AMessage(BaseModel):
    """Base A2A protocol message"""
    jsonrpc: str = "2.0"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None

class A2ACapability(BaseModel):
    """Agent capability description"""
    type: Literal["actions", "prompts", "resources", "push-notifications"]
    actions: Optional[List[str]] = None

class A2AAgentCard(BaseModel):
    """Agent card per A2A spec"""
    name: str
    description: str
    url: str
    provider: Dict[str, str]
    version: str = "1.0.0"
    defaultInputModes: List[str] = ["text", "audio"]
    defaultOutputModes: List[str] = ["text"]
    capabilities: List[A2ACapability] = []
    authentication: Optional[Dict[str, Any]] = None
    skills: List[Dict[str, Any]] = []

class A2ATask(BaseModel):
    """Task object"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: str = "task"
    name: Optional[str] = None
    inputModes: List[str] = ["text"]
    outputModes: List[str] = ["text"]
    status: Literal["submitted", "working", "input-required", "completed", "failed", "canceled"] = "submitted"
    artifacts: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class A2AClient:
    """Client for A2A protocol communication"""
    
    def __init__(self, agent_card: A2AAgentCard):
        self.agent_card = agent_card
        self.session = httpx.AsyncClient()
    
    async def send_task(self, agent_url: str, task_input: str, **kwargs) -> A2ATask:
        """Send a task to another agent"""
        message = A2AMessage(
            method="tasks/send",
            params={
                "task": {
                    "id": str(uuid.uuid4()),
                    "input": {"text": task_input},
                    "metadata": kwargs
                }
            }
        )
        
        response = await self.session.post(
            agent_url,
            json=message.model_dump(),
            timeout=300.0
        )
        
        response_data = response.json()
        return A2ATask(**response_data.get("result", {}).get("task", {}))
    
    async def get_task_status(self, agent_url: str, task_id: str) -> A2ATask:
        """Get status of a task"""
        message = A2AMessage(
            method="tasks/get",
            params={"taskId": task_id}
        )
        
        response = await self.session.post(
            agent_url,
            json=message.model_dump(),
            timeout=30.0
        )
        
        response_data = response.json()
        return A2ATask(**response_data.get("result", {}).get("task", {}))

class A2AServer:
    """Server-side A2A protocol handler"""
    
    def __init__(self, agent):
        self.agent = agent
        self.tasks: Dict[str, A2ATask] = {}
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming A2A messages"""
        a2a_msg = A2AMessage(**message)
        
        if a2a_msg.method == "agents/discover":
            return {
                "jsonrpc": "2.0",
                "id": a2a_msg.id,
                "result": {"agentCard": self.generate_agent_card()}
            }
        
        elif a2a_msg.method == "tasks/send":
            task_data = a2a_msg.params.get("task", {})
            task = A2ATask(**task_data)
            self.tasks[task.id] = task
            
            # Process task in background
            asyncio.create_task(self.process_task(task.id))
            
            return {
                "jsonrpc": "2.0",
                "id": a2a_msg.id,
                "result": {"task": task.model_dump()}
            }
        
        elif a2a_msg.method == "tasks/get":
            task_id = a2a_msg.params.get("taskId")
            task = self.tasks.get(task_id)
            if not task:
                return {
                    "jsonrpc": "2.0",
                    "id": a2a_msg.id,
                    "error": {"code": -32602, "message": "Task not found"}
                }
            
            return {
                "jsonrpc": "2.0",
                "id": a2a_msg.id,
                "result": {"task": task.model_dump()}
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": a2a_msg.id,
                "error": {"code": -32601, "message": "Method not found"}
            }
    
    def generate_agent_card(self) -> Dict[str, Any]:
        """Generate agent card for discovery"""
        return A2AAgentCard(
            name="Sidekick ADK Agent",
            description="A personal co-worker agent that helps with research, coding, and task automation",
            url="http://localhost:8000/a2a",
            provider={"name": "Capstone Project", "organization": "Personal"},
            capabilities=[
                A2ACapability(
                    type="actions",
                    actions=["research", "code_generation", "file_operations", "web_browsing"]
                )
            ],
            skills=[
                {
                    "id": "research",
                    "name": "Research Assistant",
                    "description": "Performs web research and summarizes findings"
                },
                {
                    "id": "coding",
                    "name": "Code Assistant",
                    "description": "Writes, executes, and debugs Python code"
                }
            ]
        ).model_dump()
    
    async def process_task(self, task_id: str):
        """Process a task asynchronously"""
        task = self.tasks[task_id]
        task.status = "working"
        
        try:
            # Extract input
            input_text = task.artifacts[0].get("text", "") if task.artifacts else ""
            
            # Process through sidekick
            result = await self.agent.run_superstep(
                message=input_text,
                success_criteria="Complete the requested task",
                history=[]
            )
            
            # Update task with results
            task.status = "completed"
            task.artifacts.append({
                "type": "text",
                "text": result["worker_response"]
            })
            task.artifacts.append({
                "type": "evaluation",
                "text": result["evaluator_feedback"]
            })
            
        except Exception as e:
            task.status = "failed"
            task.artifacts.append({
                "type": "error",
                "text": str(e)
            })
