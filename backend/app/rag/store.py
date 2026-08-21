import os
import re
import json
import urllib.request
import math
from typing import Dict, List, Any, Optional

try:
    import chromadb
    HAVE_CHROMADB = True
except ImportError:
    HAVE_CHROMADB = False


def semantic_chunk_resume(cand: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split candidate resume into semantic section chunks with rich metadata.
    Does NOT blindly cut every 500 characters.
    """
    chunks = []
    cand_id = str(cand.get("id", ""))
    cand_name = cand.get("name", "Candidate")
    raw_text = cand.get("raw_text", "")

    # Chunk 1: Extracted Skills Summary
    skills = cand.get("skills", [])
    tech = cand.get("technologies", [])
    lang = cand.get("languages", [])
    if skills or tech or lang:
        summary_text = (
            f"Extracted Profile Summary for Candidate {cand_name}:\n"
            f"Primary Skills: {', '.join(skills)}\n"
            f"Technologies & Tools: {', '.join(tech)}\n"
            f"Languages: {', '.join(lang)}"
        )
        chunks.append({
            "chunk_id": f"cand_{cand_id}_skills_summary",
            "candidate_id": cand_id,
            "document_type": "resume",
            "section": "skills_summary",
            "source": f"Resume ({cand_name})",
            "text": summary_text
        })

    # Chunk 2: Parse raw text by semantic section headers if present
    sections = re.split(r"\n(?=[A-Z\s]{4,}:|\n[A-Z][a-z]+\s*:)", raw_text)
    if len(sections) > 1:
        for idx, sec in enumerate(sections):
            sec_clean = sec.strip()
            if not sec_clean:
                continue
            header_match = re.match(r"^([A-Z\s]{3,20}):?", sec_clean)
            sec_title = header_match.group(1).strip().lower() if header_match else f"section_{idx+1}"
            chunks.append({
                "chunk_id": f"cand_{cand_id}_sec_{idx+1}",
                "candidate_id": cand_id,
                "document_type": "resume",
                "section": sec_title,
                "source": f"Resume ({cand_name})",
                "text": sec_clean[:1200]
            })
    else:
        # Paragraph or sentence level chunking
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 20]
        for idx, p in enumerate(paragraphs):
            chunks.append({
                "chunk_id": f"cand_{cand_id}_para_{idx+1}",
                "candidate_id": cand_id,
                "document_type": "resume",
                "section": "background_and_experience",
                "source": f"Resume ({cand_name})",
                "text": p[:1000]
            })

    return chunks


def semantic_chunk_job(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split job description into semantic requirement section chunks.
    """
    chunks = []
    job_id = str(job.get("id", ""))
    title = job.get("title", "Job Profile")
    desc = job.get("description", "")
    req_skills = job.get("required_skills", [])
    req_tech = job.get("required_technologies", [])
    pref_skills = job.get("preferred_skills", [])
    pref_tech = job.get("preferred_technologies", [])

    # Chunk 1: Job Requirements Summary
    req_text = (
        f"Job Overview & Mandatory Requirements for {title}:\n"
        f"Required Skills & Technologies: {', '.join(req_skills + req_tech)}\n"
        f"Preferred / Bonus Qualifications: {', '.join(pref_skills + pref_tech)}"
    )
    chunks.append({
        "chunk_id": f"job_{job_id}_requirements",
        "job_id": job_id,
        "document_type": "job",
        "section": "requirements",
        "source": f"Job Description (#{job_id} {title})",
        "text": req_text
    })

    # Chunk 2: Full Description
    chunks.append({
        "chunk_id": f"job_{job_id}_description",
        "job_id": job_id,
        "document_type": "job",
        "section": "full_description",
        "source": f"Job Description (#{job_id} {title})",
        "text": desc[:1500]
    })

    return chunks


class RAGVectorStore:
    """
    Persistent Vector Store using ChromaDB (or lightweight fallback embedding)
    for candidate and job document chunk index and retrieval.
    """
    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)
        self.chroma_client = None
        self.collection = None
        self.memory_store: List[Dict[str, Any]] = []

        if HAVE_CHROMADB:
            try:
                self.chroma_client = chromadb.PersistentClient(path=db_dir)
                self.collection = self.chroma_client.get_or_create_collection(
                    name="ai_recruiter_rag",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"Warning: ChromaDB initialization error: {e}")
                self.chroma_client = None

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector via Ollama nomic-embed-text API or TF-IDF fallback."""
        try:
            url = "http://localhost:11434/api/embeddings"
            payload = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                emb = data.get("embedding")
                if emb and isinstance(emb, list):
                    return emb
        except Exception:
            pass

        # Simple lightweight TF-IDF / Bag of Words Fallback (384 dims)
        words = re.findall(r"\w+", text.lower())
        vec = [0.0] * 384
        for w in words:
            idx = abs(hash(w)) % 384
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def index_candidate(self, cand: Dict[str, Any]):
        chunks = semantic_chunk_resume(cand)
        for chunk in chunks:
            self._upsert_chunk(chunk)

    def index_job(self, job: Dict[str, Any]):
        chunks = semantic_chunk_job(job)
        for chunk in chunks:
            self._upsert_chunk(chunk)

    def _upsert_chunk(self, chunk: Dict[str, Any]):
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        meta = {
            "candidate_id": str(chunk.get("candidate_id", "")),
            "job_id": str(chunk.get("job_id", "")),
            "document_type": chunk.get("document_type", ""),
            "section": chunk.get("section", ""),
            "source": chunk.get("source", ""),
        }

        # ChromaDB Upsert
        if self.collection is not None:
            try:
                emb = self._get_embedding(text)
                self.collection.upsert(
                    ids=[chunk_id],
                    embeddings=[emb],
                    documents=[text],
                    metadatas=[meta]
                )
            except Exception as e:
                print(f"ChromaDB upsert warning for {chunk_id}: {e}")

        # In-memory store fallback
        self.memory_store = [c for c in self.memory_store if c["chunk_id"] != chunk_id]
        self.memory_store.append({
            "chunk_id": chunk_id,
            "text": text,
            "metadata": meta,
            "embedding": self._get_embedding(text)
        })

    def search(
        self,
        query: str,
        candidate_id: Optional[int] = None,
        job_id: Optional[int] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search vector database for top_k relevant candidate and job chunks.
        Applies metadata filtering when candidate_id or job_id are provided.
        """
        query_emb = self._get_embedding(query)
        results = []

        # 1. Try ChromaDB Query
        if self.collection is not None:
            try:
                where_clause = {}
                if candidate_id and job_id:
                    where_clause = {"$or": [{"candidate_id": str(candidate_id)}, {"job_id": str(job_id)}]}
                elif candidate_id:
                    where_clause = {"candidate_id": str(candidate_id)}
                elif job_id:
                    where_clause = {"job_id": str(job_id)}

                kwargs: Dict[str, Any] = {
                    "query_embeddings": [query_emb],
                    "n_results": top_k,
                }
                if where_clause:
                    kwargs["where"] = where_clause

                chroma_res = self.collection.query(**kwargs)
                if chroma_res and chroma_res.get("documents") and len(chroma_res["documents"][0]) > 0:
                    docs = chroma_res["documents"][0]
                    metas = chroma_res.get("metadatas", [[]])[0]
                    distances = chroma_res.get("distances", [[0.0] * len(docs)])[0]

                    for d, m, dist in zip(docs, metas, distances):
                        results.append({
                            "text": d,
                            "section": m.get("section", "general"),
                            "source": m.get("source", "Document"),
                            "document_type": m.get("document_type", "general"),
                            "similarity_score": round(1.0 - float(dist), 4) if dist <= 1.0 else round(1.0 / (1.0 + float(dist)), 4)
                        })
                    return results
            except Exception as e:
                print(f"ChromaDB search query warning: {e}")

        # 2. Fallback: In-memory cosine similarity search
        for item in self.memory_store:
            m = item["metadata"]
            if candidate_id and m.get("candidate_id") and m["candidate_id"] != str(candidate_id):
                if not (job_id and m.get("job_id") == str(job_id)):
                    continue
            if job_id and m.get("job_id") and m["job_id"] != str(job_id):
                if not (candidate_id and m.get("candidate_id") == str(candidate_id)):
                    continue

            # Cosine similarity
            emb = item["embedding"]
            sim = sum(a * b for a, b in zip(query_emb, emb))
            results.append({
                "text": item["text"],
                "section": m.get("section", "general"),
                "source": m.get("source", "Document"),
                "document_type": m.get("document_type", "general"),
                "similarity_score": round(float(sim), 4)
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
