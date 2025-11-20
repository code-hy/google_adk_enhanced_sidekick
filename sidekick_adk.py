
import asyncio
import uuid
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

# Core ADK and GenAI Imports
from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import google.generativeai as genai
from google.genai import types as genai_types

from memory_bank import MemoryBank
from observability import ObservableAgent, traced_method, logger
from a2a_protocol import A2AClient, A2AAgentCard
import json
import pickle
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
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("FATAL: GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        
        self.session_service = InMemorySessionService()
        self.memory_bank = MemoryBank()
        
        self.tools, self.browser, self.playwright = (None, None, None)
        self.agent, self.research_agent, self.coding_agent, self.evaluator_agent = (None, None, None, None)
        self.agent_runner, self.research_runner, self.coding_runner, self.evaluator_runner = (None, None, None, None)
        
        self.session_id = str(uuid.uuid4())
        self.user_id = "default_user"
        self.app_name = "sidekick_adk"
        self.a2a_client = None
        self.state_persistence_dir = Path("./agent_states")
        self.state_persistence_dir.mkdir(exist_ok=True)
        self.pause_event = asyncio.Event()
        self.is_paused = False

    async def setup(self):
        """Initialize tools, agents, runners, and the initial session."""
        self.tools, self.browser, self.playwright = await get_all_tools()

        self.a2a_client = A2AClient(A2AAgentCard(
            name="Sidekick ADK Agent",
            description="A personal co-worker agent",
            url="http://localhost:8000",
            provider={"name": "Capstone Project"}
        ))

        research_tool_names = ["search_web", "wikipedia_lookup", "browser_navigate", "write_file", "read_file", "list_files"]
        coding_tool_names = ["execute_python", "write_file", "read_file", "list_files"]

        research_tools = [t for t in self.tools if
                         (hasattr(t, 'name') and t.name in research_tool_names) or
                         (hasattr(t, '__name__') and t.__name__ in research_tool_names)]
        coding_tools = [t for t in self.tools if
                      (hasattr(t, 'name') and t.name in coding_tool_names) or
                      (hasattr(t, '__name__') and t.__name__ in coding_tool_names)]

        # --- INSTRUCTION FIX ---
        # Add explicit instructions to always report back after tool use.
        self.research_agent = Agent(
            name="research_specialist", model="gemini-2.5-flash-lite", tools=research_tools,
            instruction="You are a research specialist. You will be given a task, success criteria, and context. " \
                        "Fulfill the task by breaking it down into steps. " \
                        "After successfully using a tool like `write_file`, you MUST provide a confirmation message to the user.",
        )
        self.coding_agent = Agent(
            name="coding_specialist", model="gemini-2.5-flash-lite", tools=coding_tools,
            instruction="You are a coding specialist. You will be given a task, success criteria, and context. " \
                        "After using a tool like `execute_python` or `write_file`, you MUST report the result or status back to the user in a final message. Do not stop silently.",
        )
        self.evaluator_agent = Agent(
            name="sidekick_evaluator", model="gemini-2.5-flash-lite",
            instruction=self._get_evaluator_instruction(),
        )
        self.agent = Agent(
            name="sidekick_orchestrator", model="gemini-2.5-flash-lite", tools=self.tools,
            instruction="You are a helpful assistant. After using a tool, always provide a final confirmation or result message to the user.",
        )

        self.research_runner = Runner(
            agent=self.research_agent, app_name=self.app_name, session_service=self.session_service
        )
        self.coding_runner = Runner(
            agent=self.coding_agent, app_name=self.app_name, session_service=self.session_service
        )
        self.evaluator_runner = Runner(
            agent=self.evaluator_agent, app_name=self.app_name, session_service=self.session_service
        )
        self.agent_runner = Runner(
            agent=self.agent, app_name=self.app_name, session_service=self.session_service
        )
        
        self._create_session()
        await self._load_state()
        self.logger.info("Sidekick ADK initialized", session_id=self.session_id)

    def _create_session(self):
        """Creates and registers a new session with the session service."""
        self.session_service.create_session(
            session_id=self.session_id, user_id=self.user_id, app_name=self.app_name,
        )
        self.logger.info("New session created in service", session_id=self.session_id)

    def _get_evaluator_instruction(self) -> str:
        """Returns the static instruction for the evaluator agent."""
        return """You are an expert evaluator. You will be given a conversation history, success criteria, and the assistant's last response.
Assess the response and provide your evaluation ONLY in valid JSON format based on the following structure:
{{
  "feedback": "detailed feedback",
  "success_criteria_met": true/false,
  "user_input_needed": true/false,
  "quality_score": 0.0-1.0
}}"""

    async def _run_agent(self, runner: Runner, message: str) -> Any:
        """Helper method to run an agent with a fully-formed prompt."""
        try:
            new_message = genai_types.Content(role="user", parts=[genai_types.Part(text=message)])
            event_generator = runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=new_message,
            )
            final_response_event = None
            async for event in event_generator:
                if event.is_final_response():
                    final_response_event = event
            return final_response_event
        except Exception as e:
            self.logger.error(f"Error running agent via runner: {e}")
            raise

    @traced_method("run_superstep")
    async def run_superstep(self, message: str, success_criteria: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run a complete worker-evaluator cycle by building full prompts."""
        if self.is_paused:
            self.logger.info("Agent is paused, waiting for resume")
            await self.pause_event.wait()

        specialist_runner = await self._route_to_specialist_runner(message)
        start_time = datetime.utcnow()

        worker_content = f"""USER REQUEST:
{message}

SUCCESS CRITERIA:
{success_criteria}

CONTEXT FROM PREVIOUS TURNS:
{self._get_feedback_context(history)}

RELEVANT MEMORIES:
{self._get_relevant_memories(message)}

The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
Please proceed with the request.
"""
        worker_response_event = None
        last_error = None
        try:
            worker_response_event = await self._run_agent(specialist_runner, worker_content)
        except Exception as e:
            last_error = e
            self.logger.error(f"Failed to run agent: {e}")

        # --- CODE FIX ---
        # This block now robustly handles the case where the worker returns None.
        if worker_response_event is None or not hasattr(worker_response_event, 'content'):
            error_content = str(last_error) if last_error else "Agent performed an action but did not provide a final response."
            last_content = f"I'm sorry, I encountered an error: {error_content}"
            self.logger.warning("Worker agent did not produce a final response event.")
        else:
            last_content = "".join(part.text for part in worker_response_event.content.parts if hasattr(part, 'text'))

        eval_result = None
        if self.evaluator_runner:
            eval_content = f"""CONVERSATION HISTORY:
{self._format_conversation(history + [{"role": "user", "content": message}])}

SUCCESS CRITERIA:
{success_criteria}

ASSISTANT'S LAST RESPONSE:
{last_content}

Please provide your evaluation in the required JSON format.
"""
            try:
                eval_response_event = await self._run_agent(self.evaluator_runner, eval_content)
                if eval_response_event:
                    eval_result = self._parse_evaluation(eval_response_event)
                else:
                    self.logger.warning("Evaluator returned no final response.")
                    eval_result = EvaluatorOutput(
                        feedback="Evaluation agent did not produce a final response.",
                        success_criteria_met=False, user_input_needed=True, quality_score=0.0
                    )
            except Exception as e:
                self.logger.error(f"Failed to run evaluator: {e}")
                eval_result = EvaluatorOutput(
                    feedback=f"Evaluation failed due to an error: {e}",
                    success_criteria_met=False, user_input_needed=True, quality_score=0.0
                )
        else:
            eval_result = EvaluatorOutput(
                feedback="Evaluation not available",
                success_criteria_met=False, user_input_needed=True, quality_score=0.5
            )

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
        duration = (datetime.utcnow() - start_time).total_seconds()
        self.logger.info(
            "superstep_completed",
            success_criteria_met=eval_result.success_criteria_met,
            quality_score=eval_result.quality_score,
            duration_seconds=duration
        )
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

    async def _route_to_specialist_runner(self, message: str) -> Runner:
        """Intelligently route to the appropriate specialist's runner"""
        coding_keywords = ["code", "python", "function", "script", "debug", "execute"]
        research_keywords = ["research", "search", "find", "lookup", "investigate"]
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in coding_keywords):
            self.logger.info("Routing to coding specialist runner")
            return self.coding_runner
        elif any(keyword in message_lower for keyword in research_keywords):
            self.logger.info("Routing to research specialist runner")
            return self.research_runner
        else:
            self.logger.info("Routing to general orchestrator runner")
            return self.agent_runner

    def _get_relevant_memories(self, query: str) -> str:
        memories = self.memory_bank.search_memories(
            query, session_id=self.session_id, limit=2, min_importance=0.7
        )
        if not memories: return "No relevant memories found."
        return "\n".join(
            f"- {m.content[:100]}..." for m in memories
        )

    def _get_feedback_context(self, history: List[Dict[str, Any]]) -> str:
        for msg in reversed(history):
            if "evaluator" in msg.get("role", "") or "feedback" in msg.get("content", "").lower():
                return f"Address this feedback from the previous turn: {msg['content']}"
        return "No specific feedback from the previous turn."

    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        conversation = ""
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            conversation += f"{role}:\n{content}\n\n"
        return conversation

    def _parse_evaluation(self, eval_response_event: Any) -> EvaluatorOutput:
        if eval_response_event is None or not hasattr(eval_response_event, 'content'):
            return EvaluatorOutput(
                feedback="Evaluator did not produce a response.",
                success_criteria_met=False, user_input_needed=True, quality_score=0.0
            )
        content = "".join(part.text for part in eval_response_event.content.parts if hasattr(part, 'text'))
        try:
            json_str = content[content.find('{') : content.rfind('}')+1]
            data = json.loads(json_str)
            return EvaluatorOutput(**data)
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse JSON from evaluator response: {e}. Content: {content}")
            success_criteria_met = "success_criteria_met\": true" in content.lower()
            user_input_needed = "user_input_needed\": true" in content.lower()
            return EvaluatorOutput(
                feedback=f"Could not parse JSON, but got this text: {content}",
                success_criteria_met=success_criteria_met,
                user_input_needed=user_input_needed,
                quality_score=0.1
            )

    async def pause(self):
        self.is_paused = True
        self.pause_event.clear()
        self.logger.info("Agent paused")
        await self._persist_state()

    def resume(self):
        self.is_paused = False
        self.pause_event.set()
        self.logger.info("Agent resumed")

    async def _persist_state(self):
        state_file = self.state_persistence_dir / f"{self.session_id}.pkl"
        state = {"session_id": self.session_id, "user_id": self.user_id, "timestamp": datetime.utcnow().isoformat(),
                 "memory_bank_stats": {"total_memories": len(self.memory_bank.memories)}}
        with open(state_file, 'wb') as f: pickle.dump(state, f)
        self.logger.info("State persisted", state_file=str(state_file))

    async def _load_state(self):
        state_file = self.state_persistence_dir / f"{self.session_id}.pkl"
        if state_file.exists():
            try:
                with open(state_file, 'rb') as f: state = pickle.load(f)
                self.session_id = state.get("session_id", self.session_id)
                self._create_session()
                self.logger.info("State loaded and session re-created", state_file=str(state_file))
            except Exception as e:
                self.logger.error("Failed to load state", error=str(e))

    async def reset(self):
        await self.cleanup()
        self.session_id = str(uuid.uuid4())
        self._create_session()
        self.logger.info("Session reset", new_session_id=self.session_id)

    async def cleanup(self):
        if self.browser:
            try: await self.browser.close()
            except: pass
        if self.playwright:
            try: await self.playwright.stop()
            except: pass
        self.logger.info("Resources cleaned up")