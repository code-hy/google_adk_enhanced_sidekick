# memory_bank.py - Pure Python Version (No ChromaDB needed)

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import pickle

# Simple memory storage for development
class MemoryBank:
    """Lightweight in-memory storage with basic search capabilities."""
    
    def __init__(self, persist_directory: str = "./memory_bank"):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(exist_ok=True)
        self.memories: List[Dict[str, Any]] = []
        self.load_from_disk()  # Load saved memories on startup
        print(f"💾 MemoryBank initialized (simple storage)")
        
    def add_memory(
        self,
        content: str,
        session_id: str,
        metadata: Dict[str, Any] = None,
        importance_score: float = 0.5
    ) -> str:
        """Add a new memory entry."""
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        memory = {
            "id": memory_id,
            "content": content[:500],  # Limit size
            "metadata": metadata or {},
            "timestamp": timestamp,
            "session_id": session_id,
            "importance_score": importance_score
        }
        
        self.memories.append(memory)
        self.save_to_disk()  # Persist immediately
        return memory_id
    
    def search_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 5,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search memories by keyword matching (simple but effective)."""
        # Filter by importance and session
        candidates = [
            m for m in self.memories 
            if m["importance_score"] >= min_importance
            and (session_id is None or m["session_id"] == session_id)
        ]
        
        # Simple keyword matching
        query_words = query.lower().split()
        scored = []
        
        for memory in candidates:
            content_lower = memory["content"].lower()
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                scored.append((memory, score))
        
        # Sort by relevance and importance
        scored.sort(key=lambda x: (x[1], x[0]["importance_score"]), reverse=True)
        return [item[0] for item in scored[:limit]]
    
    def get_session_summary(self, session_id: str) -> str:
        """Generate a summary of a session."""
        session_memories = [m for m in self.memories if m["session_id"] == session_id]
        
        if not session_memories:
            return "No memories found for this session."
        
        summary = f"Session {session_id} Summary:\n"
        summary += "=" * 50 + "\n"
        summary += f"Total memories: {len(session_memories)}\n\n"
        
        # Show top memories
        for i, memory in enumerate(sorted(session_memories, key=lambda x: x["importance_score"], reverse=True)[:5], 1):
            summary += f"{i}. Score: {memory['importance_score']:.2f}\n"
            summary += f"   {memory['content'][:120]}...\n"
        
        return summary
    
    def compact_context(self, full_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Intelligently compact context window."""
        # Keep recent messages
        recent = full_history[-8:]  # Last 8 messages
        
        # Add relevant memories if space
        if len(full_history) > 0:
            query = full_history[-1].get("content", "")
            important_memories = self.search_memories(query, limit=2, min_importance=0.7)
            
            if important_memories:
                memory_context = {
                    "role": "system",
                    "content": "Relevant memories:\n" + "\n".join(
                        f"- {m['content'][:100]}..." for m in important_memories
                    )
                }
                return [memory_context] + recent
        
        return recent
    
    def save_to_disk(self):
        """Persist memories to disk."""
        try:
            file_path = self.persist_dir / "memories.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(self.memories, f)
        except Exception as e:
            print(f"⚠️ Failed to save memories: {e}")
    
    def load_from_disk(self):
        """Load memories from disk."""
        try:
            file_path = self.persist_dir / "memories.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    self.memories = pickle.load(f)
                print(f"💾 Loaded {len(self.memories)} memories from disk")
        except Exception as e:
            print(f"⚠️ Failed to load memories: {e}")
            self.memories = []

# Optional: If you want vector search later, install these:
# pip install chromadb>=0.4.0 sentence-transformers