import os
import uuid
from pathlib import Path
from typing import List, Dict

import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from feast import FeatureStore

# Ensure ROOT and Feast paths
ROOT = Path(__file__).resolve().parent.parent
FEAST_DIR = ROOT / "app" / "feast_repo"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
COLLECTION_NAME = "personal_episodic_memory"

class HybridMemoryAgent:
    def __init__(self):
        # 1. Initialize OpenAI and Qdrant
        self.openai_client = openai.OpenAI()
        self.qdrant_client = QdrantClient(":memory:")
        
        # Create Qdrant Collection if not exists
        existing = {c.name for c in self.qdrant_client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            
        # 2. Initialize In-memory BM25 Store (Simplified for POC)
        self.memories_db: List[Dict] = []
        self.bm25: BM25Okapi = None
        
        # 3. Initialize Feast Feature Store
        self.fs = FeatureStore(repo_path=str(FEAST_DIR))
        
        # Default Features to fetch
        self.profile_features = [
            "user_profile_features:topic_affinity",
            "user_profile_features:reading_speed_wpm",
            "query_velocity_features:queries_last_hour"
        ]

    def _tokenize(self, text: str) -> List[str]:
        """Use Pyvi for robust Vietnamese tokenization."""
        return ViTokenizer.tokenize(text).lower().split()

    def _update_bm25(self):
        if not self.memories_db:
            return
        tokenized = [self._tokenize(doc["text"]) for doc in self.memories_db]
        self.bm25 = BM25Okapi(tokenized)

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        doc_id = str(uuid.uuid4())
        
        # 1. Embed text using OpenAI
        response = self.openai_client.embeddings.create(input=[text], model=EMBED_MODEL)
        vector = response.data[0].embedding
        
        # 2. Upsert to Qdrant with user_id payload for isolation
        self.qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=len(self.memories_db) + 1,
                    vector=vector,
                    payload={"doc_id": doc_id, "user_id": user_id, "text": text}
                )
            ]
        )
        
        # 3. Add to local memory DB for BM25
        self.memories_db.append({"doc_id": doc_id, "user_id": user_id, "text": text})
        self._update_bm25()
        print(f"[Memory Added] Stored for {user_id}: '{text[:30]}...'")

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top-K memories + user profile features → return assembled context."""
        # --- 1. Get User Profile from Feast Feature Store ---
        try:
            profile = self.fs.get_online_features(
                features=self.profile_features,
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            
            topic = profile.get("topic_affinity", ["unknown"])[0]
            speed = profile.get("reading_speed_wpm", [0])[0]
            recent_queries = profile.get("queries_last_hour", [0])[0]
        except Exception as e:
            # Fallback if Feast is not materialized
            topic, speed, recent_queries = "unknown", 0, 0
            
        # --- 2. Hybrid Search (Qdrant Semantic + In-memory BM25) ---
        semantic_hits = []
        kw_hits = []
        
        if self.memories_db:
            # Semantic Search filtered by user_id
            q_vec = self.openai_client.embeddings.create(input=[query], model=EMBED_MODEL).data[0].embedding
            sem_results = self.qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=q_vec,
                query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
                limit=top_k * 2
            )
            semantic_hits = [p.payload for p in sem_results.points]
            
            # Keyword Search (BM25) filtered by user_id
            if self.bm25:
                scores = self.bm25.get_scores(self._tokenize(query))
                ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])
                for idx in ranked_indices:
                    doc = self.memories_db[idx]
                    if doc["user_id"] == user_id:
                        kw_hits.append(doc)
                    if len(kw_hits) >= top_k * 2:
                        break
        
        # Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        scores_map = {}
        meta_map = {}
        for hits in (semantic_hits, kw_hits):
            for rank, hit in enumerate(hits, start=1):
                doc_id = hit["doc_id"]
                scores_map[doc_id] = scores_map.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
                meta_map[doc_id] = hit["text"]
                
        ordered_memories = sorted(scores_map.items(), key=lambda kv: -kv[1])[:top_k]
        memory_texts = [f"- {meta_map[doc_id]}" for doc_id, _ in ordered_memories]
        
        if not memory_texts:
            memory_texts = ["- (No relevant episodic memory found)"]
            
        # --- 3. Assemble Context String ---
        context = f"""
=================================================
SYSTEM CONTEXT FOR LLM (User: {user_id})
=================================================
[STABLE PROFILE]
- Topic Affinity : {topic}
- Reading Speed  : {speed} wpm

[RECENT ACTIVITY]
- Queries in last hour: {recent_queries}

[EPISODIC MEMORY (Top-{top_k} Hybrid Hits)]
{chr(10).join(memory_texts)}
=================================================
"""
        return context
