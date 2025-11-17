import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import os
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    """Represents a memory entry"""
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    session_id: str
    importance_score: float = 0.5

class MemoryBank:
    """Long-term memory system with vector search"""
    
    def __init__(self, persist_directory: str = "./memory_bank"):
        self.persist_dir = persist_directory
        os.makedirs(persist_dir, exist_ok=True)
        
        # ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="sidekick_memories",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Context compaction threshold
        self.max_context_tokens = 8000
        self.importance_threshold = 0.6
    
    def add_memory(
        self,
        content: str,
        session_id: str,
        metadata: Dict[str, Any] = None,
        importance_score: float = 0.5
    ) -> str:
        """Add a new memory entry"""
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # Create embedding
        embedding = self.embedder.encode(content).tolist()
        
        # Prepare metadata
        meta = {
            "session_id": session_id,
            "timestamp": timestamp.isoformat(),
            "importance_score": importance_score,
            **(metadata or {})
        }
        
        # Store in ChromaDB
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta]
        )
        
        return memory_id
    
    def search_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 5,
        min_importance: float = 0.0
    ) -> List[MemoryEntry]:
        """Search memories by semantic similarity"""
        query_embedding = self.embedder.encode(query).tolist()
        
        # Build filter
        where_filter = {"importance_score": {"$gte": min_importance}}
        if session_id:
            where_filter["session_id"] = session_id
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
            include=["documents", "metadatas"]
        )
        
        memories = []
        if results["ids"]:
            for i, memory_id in enumerate(results["ids"][0]):
                memories.append(
                    MemoryEntry(
                        id=memory_id,
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        timestamp=datetime.fromisoformat(
                            results["metadatas"][0][i]["timestamp"]
                        ),
                        session_id=results["metadatas"][0][i]["session_id"],
                        importance_score=results["metadatas"][0][i]["importance_score"]
                    )
                )
        
        return memories
    
    def get_session_summary(self, session_id: str) -> str:
        """Generate a summary of a session"""
        results = self.collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            return "No memories found for this session."
        
        # Sort by importance
        memories = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: x[1].get("importance_score", 0),
            reverse=True
        )
        
        # Generate summary
        summary = f"Session {session_id} Summary:\n"
        summary += "=" * 50 + "\n"
        
        for i, (doc, meta) in enumerate(memories[:10], 1):
            summary += f"\n{i}. {doc[:100]}..."
            summary += f"\n   Importance: {meta.get('importance_score', 0):.2f}"
        
        return summary
    
    def compact_context(self, full_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Intelligently compact context window"""
        # Calculate token count (rough estimation)
        def estimate_tokens(text: str) -> int:
            return len(text.split()) * 1.3  # Rough estimate
        
        current_tokens = sum(
            estimate_tokens(msg.get("content", ""))
            for msg in full_history
        )
        
        if current_tokens <= self.max_context_tokens:
            return full_history
        
        # Keep recent messages
        compacted = []
        token_budget = self.max_context_tokens * 0.7  # 70% for recent
        
        # Add system messages first
        for msg in full_history:
            if msg.get("role") == "system":
                compacted.append(msg)
                token_budget -= estimate_tokens(msg.get("content", ""))
        
        # Add recent messages from end
        for msg in reversed(full_history):
            if msg.get("role") != "system":
                msg_tokens = estimate_tokens(msg.get("content", ""))
                if token_budget - msg_tokens > 0:
                    compacted.insert(len([m for m in compacted if m.get("role") == "system"]), msg)
                    token_budget -= msg_tokens
                else:
                    break
        
        # Add memory-augmented context
        if full_history:
            recent_query = full_history[-1].get("content", "")
            important_memories = self.search_memories(
                recent_query,
                limit=3,
                min_importance=self.importance_threshold
            )
            
            if important_memories:
                memory_context = {
                    "role": "system",
                    "content": "Relevant past memories:\n" + "\n".join(
                        f"- {m.content[:150]}..." for m in important_memories
                    )
                }
                compacted.insert(len([m for m in compacted if m.get("role") == "system"]), memory_context)
        
        return compacted
