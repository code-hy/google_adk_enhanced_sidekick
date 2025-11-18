import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path 
from pydantic import BaseModel, Field
from google.adk import Agent
from google.adk.sessions import InMemorySessionService
from google.genai import types
from memory_bank import MemoryBank
from observability import ObservableAgent, traced_method, logger
from a2a_protocol import A2AClient, A2AAgentCard
import json
import pickle
import os
from sidekick_tools_adk import get_all_tools

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )
    quality_score: float = Field(
        default=0.0,
        description="Quality score of the response from 0.0 to 1.0"
    )

class SidekickADK(ObservableAgent):
    def __init__(self):
        ObservableAgent.__init__(self, "sidekick_orchestrator")
        self.session_service = InMemorySessionService()
        self.memory_bank = MemoryBank()
        self.tools = None
        self.browser = None
        self.playwright = None
        self.agent = None
        self.session_id = str(uuid.uuid4())
        self.user_id = "default_user"
        self.a2a_client = None
        self.state_persistence_dir = Path("./agent_states")
        self.state_persistence_dir.mkdir(exist_ok=True)
        self.pause_event = asyncio.Event()
        self.is_paused = False
        
    async def setup(self):
        """Initialize tools and agent"""
        self.tools, self.browser, self.playwright = await get_all_tools()
        
        # Add A2A capability
        self.a2a_client = A2AClient(A2AAgentCard(
            name="Sidekick ADK Agent",
            description="A personal co-worker agent",
            url="http://localhost:8000",
            provider={"name": "Capstone Project"}
        ))
        
        # Create specialized sub-agents
        self.research_agent = Agent(
            name="research_specialist",
            model="gemini-1.5-pro",  # More powerful for research
            tools=[t for t in self.tools if t.name in ["search_web", "wikipedia_lookup", "browser_navigate"]],
            instruction="You are a research specialist. Find relevant, accurate information and cite sources.",
        )
        
        self.coding_agent = Agent(
            name="coding_specialist",
            model="gemini-1.5-flash",
            tools=[t for t in self.tools if t.name in ["execute_python", "write_file", "read_file"]],
            instruction="You are a coding specialist. Write clean, efficient code with proper error handling.",
        )
        
        self.evaluator_agent = Agent(
            name="sidekick_evaluator",
            model="gemini-1.5-flash",
            instruction=self._get_evaluator_instruction(),
        )
        
        # Main orchestrator
        self.agent = Agent(
            name="sidekick_orchestrator",
            model="gemini-1.5-flash",
            instruction="You orchestrate the sidekick workflow. Route tasks to specialists and manage the evaluation loop.",
            agents=[self.research_agent, self.coding_agent, self.evaluator_agent],
            tools=self.tools
        )
        
        # Load persisted state if exists
        await self._load_state()
        
        self.logger.info("Sidekick ADK initialized", session_id=self.session_id)
        
    def _get_worker_instruction(self) -> str:
        return f"""You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
You have many tools to help you, including tools to browse the internet, navigate and retrieve web pages.
You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This is the success criteria: {{success_criteria}}
You should reply either with a question for the user about this assignment, or with your final response.
{{feedback_context}}
{{memory_context}}"""

    def _get_evaluator_instruction(self) -> str:
        return """You are an expert evaluator that determines if a task has been completed successfully.
Assess the response on:
1. Accuracy (0-10)
2. Completeness (0-10)
3. Clarity (0-10)
4. Success criteria alignment (0-10)

Calculate an overall quality score (0-1.0).
Provide detailed feedback and determine if success criteria is met.
Decide if user input is needed.

Conversation: {conversation}
Success criteria: {success_criteria}
Last response: {last_response}

Respond in JSON:
{{
  "feedback": "detailed feedback",
  "success_criteria_met": true/false,
  "user_input_needed": true/false,
  "quality_score": 0.0-1.0
}}"""
    
    @traced_method("run_superstep")
    async def run_superstep(self, message: str, success_criteria: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run a complete worker-evaluator cycle with pause/resume and memory"""
        
        # Check if paused
        if self.is_paused:
            self.logger.info("Agent is paused, waiting for resume")
            await self.pause_event.wait()
        
        session = self.session_service.get_session(
            app_name="sidekick",
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        # Compact context and add memories
        compacted_history = self.memory_bank.compact_context(history)
        
        # Route to appropriate specialist
        specialist = await self._route_to_specialist(message)
        
        # Prepare context
        memory_context = self._get_relevant_memories(message)
        feedback_context = self._get_feedback_context(history)
        
        # Execute with specialist
        start_time = datetime.utcnow()
        
        worker_content = f"""User request: {message}
Success criteria: {success_criteria}
{feedback_context}
{memory_context}"""
        
        # Run specialist agent
        worker_response = await specialist.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            message=worker_content,
            app_name="sidekick"
        )
        
        last_content = worker_response.content if hasattr(worker_response, 'content') else str(worker_response)
        
        # Evaluate response
        eval_content = self._get_evaluator_instruction().format(
            conversation=self._format_conversation(history + [{"role": "user", "content": message}]),
            success_criteria=success_criteria,
            last_response=last_content
        )
        
        eval_response = await self.evaluator_agent.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            message=eval_content,
            app_name="sidekick"
        )
        
        # Parse evaluation
        eval_result = self._parse_evaluation(eval_response)
        
        # Store memory
        importance_score = eval_result.quality_score
        self.memory_bank.add_memory(
            content=f"Task: {message}\nResponse: {last_content}\nEvaluation: {eval_result.feedback}",
            session_id=self.session_id,
            metadata={
                "type": "task_execution",
                "success_criteria_met": eval_result.success_criteria_met,
                "quality_score": eval_result.quality_score
            },
            importance_score=importance_score
        )
        
        # Log metrics
        duration = (datetime.utcnow() - start_time).total_seconds()
        session_duration.record(duration, {"session_id": self.session_id})
        evaluation_score_histogram.record(eval_result.quality_score)
        
        self.logger.info(
            "superstep_completed",
            success_criteria_met=eval_result.success_criteria_met,
            quality_score=eval_result.quality_score,
            duration_seconds=duration
        )
        
        # Persist state if needed
        if eval_result.user_input_needed or eval_result.success_criteria_met:
            await self._persist_state()
        
        return {
            "worker_response": last_content,
            "evaluator_feedback": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
            "quality_score": eval_result.quality_score,
            "conversation_update": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": last_content},
                {"role": "evaluator", "content": f"Score: {eval_result.quality_score:.2f} - {eval_result.feedback}"}
            ]
        }
    
    async def _route_to_specialist(self, message: str) -> Agent:
        """Intelligently route to appropriate specialist agent"""
        # Simple routing logic - can be enhanced with LLM
        coding_keywords = ["code", "python", "function", "script", "debug"]
        research_keywords = ["research", "search", "find", "lookup", "investigate"]
        
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in coding_keywords):
            self.logger.info("Routing to coding specialist")
            return self.coding_agent
        elif any(keyword in message_lower for keyword in research_keywords):
            self.logger.info("Routing to research specialist")
            return self.research_agent
        else:
            return self.worker_agent
    
    def _get_relevant_memories(self, query: str) -> str:
        """Retrieve relevant memories for context"""
        memories = self.memory_bank.search_memories(
            query,
            session_id=self.session_id,
            limit=2,
            min_importance=0.7
        )
        
        if not memories:
            return ""
        
        return "\n\nRelevant past context:\n" + "\n".join(
            f"- {m.content[:100]}..." for m in memories
        )
    
    def _get_feedback_context(self, history: List[Dict[str, Any]]) -> str:
        """Extract feedback context from history"""
        for msg in reversed(history):
            if "evaluator" in msg.get("role", "") or "feedback" in msg.get("content", "").lower():
                return f"\nPrevious feedback: {msg['content']}\nPlease address this feedback."
        return ""
    
    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Format conversation for evaluator"""
        conversation = "Conversation history:\n\n"
        for msg in messages:
            if msg.get("role") == "user":
                conversation += f"User: {msg['content']}\n"
            elif msg.get("role") == "assistant":
                conversation += f"Assistant: {msg['content'][:200]}...\n"
        return conversation
    
    def _parse_evaluation(self, eval_response: Any) -> EvaluatorOutput:
        """Parse evaluator response into structured format"""
        content = eval_response.content if hasattr(eval_response, 'content') else str(eval_response)
        
        # Try to parse JSON
        try:
            if "{" in content:
                json_start = content.index("{")
                json_end = content.rindex("}") + 1
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                return EvaluatorOutput(**data)
        except:
            pass
        
        # Fallback to simple parsing
        success_criteria_met = "success criteria met" in content.lower() and "not met" not in content.lower()
        user_input_needed = "user input needed" in content.lower() or "question" in content.lower()
        
        # Extract quality score if present
        quality_score = 0.5
        if "score:" in content.lower():
            try:
                import re
                score_match = re.search(r"score[: ]*(\d+(\.\d+)?)", content.lower())
                if score_match:
                    quality_score = float(score_match.group(1)) / 10
            except:
                pass
        
        return EvaluatorOutput(
            feedback=content,
            success_criteria_met=success_criteria_met,
            user_input_needed=user_input_needed,
            quality_score=min(quality_score, 1.0)
        )
    
    async def pause(self):
        """Pause the agent"""
        self.is_paused = True
        self.pause_event.clear()
        self.logger.info("Agent paused")
        await self._persist_state()
    
    def resume(self):
        """Resume the agent"""
        self.is_paused = False
        self.pause_event.set()
        self.logger.info("Agent resumed")
    
    async def _persist_state(self):
        """Persist agent state to disk for resume capability"""
        state_file = self.state_persistence_dir / f"{self.session_id}.pkl"
        
        state = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "memory_bank_stats": {
                "total_memories": len(self.memory_bank.collection.get()["ids"])
            }
        }
        
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info("State persisted", state_file=str(state_file))
    
    async def _load_state(self):
        """Load persisted state if exists"""
        state_file = self.state_persistence_dir / f"{self.session_id}.pkl"
        
        if state_file.exists():
            try:
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                self.logger.info("State loaded", state_file=str(state_file))
            except Exception as e:
                self.logger.error("Failed to load state", error=str(e))
    
    async def reset(self):
        """Reset the sidekick session"""
        await self.cleanup()
        self.session_id = str(uuid.uuid4())
        await self.setup()
        self.logger.info("Session reset", new_session_id=self.session_id)
    
    async def cleanup(self):
        """Clean up resources"""
        if self.browser:
            try:
                await self.browser.close()
            except:
                pass
        if self.playwright:
            try:
                await self.playwright.stop()
            except:
                pass
        await mcp_manager.disconnect_all()
        self.logger.info("Resources cleaned up")
