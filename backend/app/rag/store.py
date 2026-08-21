import os
import re
import json
import urllib.request
import math
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

try:
    import chromadb
    HAVE_CHROMADB = True
except ImportError:
    HAVE_CHROMADB = False


# ---------------------------------------------------------------------------
# Project-aware section headers used to detect PROJECTS section in resumes
# ---------------------------------------------------------------------------
_ALL_SECTION_HEADERS = re.compile(
    r"^\s*(?:PROJECTS?|PERSONAL\s+PROJECTS?|KEY\s+PROJECTS?|NOTABLE\s+PROJECTS?|ENGINEERING\s+PROJECTS?|SIDE\s+PROJECTS?|"
    r"EXPERIENCE|WORK\s+EXPERIENCE|EDUCATION|SKILLS?|TECHNOLOGIES?|LANGUAGES?|CERTIFICATIONS?|SUMMARY|OBJECTIVE|"
    r"PUBLICATIONS?|AWARDS?|ACHIEVEMENTS?|CONTACT)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_resume_into_sections(raw_text: str) -> List[Dict[str, str]]:
    """Split raw resume text into named sections."""
    lines = raw_text.splitlines()
    sections: List[Dict[str, str]] = []
    current_header = "preamble"
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        is_header = bool(_ALL_SECTION_HEADERS.match(stripped)) if stripped else False
        # Also detect ALL-CAPS lines of 3+ chars as section dividers
        if not is_header and stripped and stripped.isupper() and len(stripped) > 3:
            is_header = True

        if is_header and stripped:
            if current_lines:
                sections.append({"header": current_header, "content": "\n".join(current_lines).strip()})
            current_header = stripped.lower().rstrip(":")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"header": current_header, "content": "\n".join(current_lines).strip()})

    return sections


def _extract_project_chunks(section_content: str, cand_id: str, cand_name: str) -> List[Dict[str, Any]]:
    """
    Given the content of a PROJECTS section, extract individual projects as atomic chunks.
    Each project chunk preserves: project name + description + technologies + metrics.
    Tries two splitting strategies:
      1. Double-newline separated blocks
      2. Lines that look like project titles (non-bullet, non-indented)
    """
    chunks = []
    # Strategy 1: double-newline separated blocks
    project_blocks = re.split(r"\n{2,}", section_content)

    if len(project_blocks) <= 1:
        # Strategy 2: split on lines that are not indented/bulleted (project title lines)
        project_blocks = []
        current: List[str] = []
        for line in section_content.splitlines():
            # A line that starts a new project: non-empty, not a bullet, not indented
            if line and not line.startswith((" ", "\t", "-", "•", "*")) and current:
                project_blocks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            project_blocks.append("\n".join(current))

    for idx, block in enumerate(project_blocks):
        block = block.strip()
        if not block or len(block) < 20:
            continue

        lines = [l.strip() for l in block.splitlines() if l.strip()]
        project_name = lines[0] if lines else f"Project {idx+1}"
        project_name = re.sub(r"^[-•*\s]+", "", project_name).strip()

        if block.strip().startswith(project_name):
            chunk_text = block.strip()
        else:
            chunk_text = f"{project_name}\n{block.strip()}"

        chunks.append({
            "chunk_id": f"cand_{cand_id}_project_{idx+1}",
            "candidate_id": cand_id,
            "document_type": "resume",
            "section": "projects",
            "project_name": project_name,
            "source": f"Resume ({cand_name})",
            "text": chunk_text[:1200],
        })

    return chunks


def _normalize_section_header(header: str) -> str:
    """Normalize raw section headers to standard taxonomy sections."""
    h = header.lower().strip().rstrip(":")
    if any(kw in h for kw in ["project", "personal project", "key project", "notable project", "engineering project", "side project"]):
        return "projects"
    if any(kw in h for kw in ["education", "academic", "degree", "university", "college", "school"]):
        return "education"
    if any(kw in h for kw in ["work experience", "experience", "employment", "work history", "career", "professional experience"]):
        return "experience"
    if any(kw in h for kw in ["certif", "license", "credential", "accreditation"]):
        return "certifications"
    if any(kw in h for kw in ["skill", "technolog", "competenc", "stack", "tool"]):
        return "skills"
    if any(kw in h for kw in ["summary", "profile", "objective", "about", "bio"]):
        return "summary"
    return h or "general"


def semantic_chunk_resume(cand: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split candidate resume into semantic section chunks with rich metadata.

    Key design decisions:
    - Project sections are preserved as ATOMIC units: project name + description +
      technologies stay together in one chunk.
    - Each chunk carries normalized section and project_name metadata for downstream filtering & reranking.
    - skills_summary chunk is always created from extracted structured data.
    """
    chunks = []
    cand_id = str(cand.get("id", ""))
    cand_name = cand.get("name", "Candidate")
    raw_text = cand.get("raw_text", "")

    # --- Chunk 0: Extracted Skills/Technology Summary ---
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
            "project_name": "",
            "source": f"Resume ({cand_name})",
            "text": summary_text,
        })

    # --- Parse raw_text into sections ---
    sections = _split_resume_into_sections(raw_text)
    produced_project_chunks = False

    for sec_idx, sec in enumerate(sections):
        raw_header = sec["header"]
        norm_section = _normalize_section_header(raw_header)
        content = sec["content"].strip()
        if not content or len(content) < 10:
            continue

        if norm_section == "projects":
            project_chunks = _extract_project_chunks(content, cand_id, cand_name)
            if project_chunks:
                chunks.extend(project_chunks)
                produced_project_chunks = True
            else:
                chunks.append({
                    "chunk_id": f"cand_{cand_id}_sec_projects",
                    "candidate_id": cand_id,
                    "document_type": "resume",
                    "section": "projects",
                    "project_name": "",
                    "source": f"Resume ({cand_name})",
                    "text": content[:1200],
                })
                produced_project_chunks = True
        else:
            chunks.append({
                "chunk_id": f"cand_{cand_id}_sec_{sec_idx+1}_{norm_section[:20]}",
                "candidate_id": cand_id,
                "document_type": "resume",
                "section": norm_section,
                "project_name": "",
                "source": f"Resume ({cand_name})",
                "text": content[:1200],
            })

    # Fallback: if raw_text yielded no meaningful sections beyond skills_summary
    if len(chunks) <= 1 and raw_text:
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 20]
        for idx, p in enumerate(paragraphs):
            chunks.append({
                "chunk_id": f"cand_{cand_id}_para_{idx+1}",
                "candidate_id": cand_id,
                "document_type": "resume",
                "section": "background_and_experience",
                "project_name": "",
                "source": f"Resume ({cand_name})",
                "text": p[:1000],
            })

    return chunks


def semantic_chunk_job(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split job description into semantic requirement section chunks."""
    chunks = []
    job_id = str(job.get("id", ""))
    title = job.get("title", "Job Profile")
    desc = job.get("description", "")
    req_skills = job.get("required_skills", [])
    req_tech = job.get("required_technologies", [])
    pref_skills = job.get("preferred_skills", [])
    pref_tech = job.get("preferred_technologies", [])

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
        "project_name": "",
        "source": f"Job Description (#{job_id} {title})",
        "text": req_text,
    })

    if desc:
        chunks.append({
            "chunk_id": f"job_{job_id}_description",
            "job_id": job_id,
            "document_type": "job",
            "section": "full_description",
            "project_name": "",
            "source": f"Job Description (#{job_id} {title})",
            "text": desc[:1500],
        })

    return chunks


# ---------------------------------------------------------------------------
# Deterministic Reranker
# ---------------------------------------------------------------------------

def rerank_chunks(
    chunks: List[Dict[str, Any]],
    query_entities: List[str],
    intent: str,
) -> List[Dict[str, Any]]:
    """
    Deterministic intent-aware reranker applied AFTER cosine retrieval.

    Scoring model (additive):
      base_score         = cosine similarity from ChromaDB / in-memory
      entity_bonus       = +0.15 per query entity found in chunk text (capped at +0.45)
      project_name_bonus = +0.30-0.40 if query entity matches chunk project_name
      section_bonus      = intent-specific section weights and penalties
    """
    q_entities_lower = [e.lower() for e in query_entities if e]

    for chunk in chunks:
        score = float(chunk.get("similarity_score", 0.0))
        text_lower = chunk.get("text", "").lower()
        section = chunk.get("section", "").lower()
        project_name = chunk.get("project_name", "").lower()

        # Entity match bonus
        entity_hits = sum(1 for e in q_entities_lower if e in text_lower)
        entity_bonus = min(entity_hits * 0.15, 0.45)

        # Project name exact match bonus
        proj_name_bonus = 0.0
        if project_name and any(e and e in project_name for e in q_entities_lower):
            proj_name_bonus = 0.40 if intent in ("PROJECT_DETAIL", "CONTEXTUAL_PROJECT_QUERY") else 0.30

        # Section bonuses / penalties based on intent
        section_bonus = 0.0

        if intent in ("CANDIDATE_PROJECTS", "CANDIDATE_PROJECT"):
            if section == "projects":
                section_bonus = 0.30
            elif section in ("experience", "work experience", "background_and_experience"):
                section_bonus = 0.05
            elif section == "skills_summary":
                section_bonus = -0.20
            elif section == "education":
                section_bonus = -0.20

        elif intent in ("PROJECT_DETAIL", "CONTEXTUAL_PROJECT_QUERY"):
            if section == "projects":
                section_bonus = 0.30
            elif section == "skills_summary":
                section_bonus = -0.20
            elif section == "education":
                section_bonus = -0.20

        elif intent in ("CANDIDATE_EDUCATION",):
            if section == "education":
                section_bonus = 0.40
            elif section in ("experience", "work experience"):
                section_bonus = 0.05
            elif section == "skills_summary":
                section_bonus = -0.25
            elif section == "projects":
                section_bonus = -0.20

        elif intent in ("CANDIDATE_SKILLS", "CANDIDATE_TECHNOLOGIES", "CANDIDATE_LANGUAGES"):
            if section in ("skills_summary", "skills"):
                section_bonus = 0.30
            elif section in ("experience", "work experience", "background_and_experience"):
                section_bonus = 0.15
            elif section == "projects":
                section_bonus = 0.10
            elif section == "education":
                section_bonus = -0.20

        elif intent in ("CANDIDATE_EMPLOYMENT", "CANDIDATE_TENURE", "CANDIDATE_EXPERIENCE"):
            if section in ("experience", "work experience", "background_and_experience"):
                section_bonus = 0.35
            elif section == "skills_summary":
                section_bonus = -0.15
            elif section == "projects":
                section_bonus = 0.05
            elif section == "education":
                section_bonus = -0.10

        elif intent in ("CANDIDATE_CERTIFICATIONS",):
            if section == "certifications":
                section_bonus = 0.40
            elif section in ("skills_summary", "skills"):
                section_bonus = 0.15
            elif section in ("experience", "work experience"):
                section_bonus = 0.05
            elif section == "projects":
                section_bonus = -0.15

        elif intent in ("JOB_REQUIREMENTS", "JOB_TECHNOLOGIES", "JOB_PREFERRED", "MISSING_SKILLS_GAPS", "MISSING_SKILLS"):
            if section in ("requirements", "skills_summary"):
                section_bonus = 0.35
            elif section == "full_description":
                section_bonus = 0.10

        elif intent in ("JOB_DESCRIPTION", "JOB_RESPONSIBILITIES"):
            if section == "full_description":
                section_bonus = 0.35
            elif section == "requirements":
                section_bonus = 0.15

        elif intent in ("CANDIDATE_FIT", "SCORE_EXPLANATION", "MATCH_EVIDENCE", "COMPARISON", "CANDIDATE_SKILLS_EVIDENCE", "CANDIDATE_FIT_EVIDENCE"):
            if section in ("requirements", "skills_summary"):
                section_bonus = 0.25
            elif section in ("projects", "experience"):
                section_bonus = 0.15

        chunk["entity_score"] = round(entity_bonus + proj_name_bonus, 4)
        chunk["section_score"] = round(section_bonus, 4)
        chunk["rerank_score"] = round(score + entity_bonus + proj_name_bonus + section_bonus, 4)

    chunks.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
    return chunks


# ---------------------------------------------------------------------------
# RAGVectorStore
# ---------------------------------------------------------------------------

class RAGVectorStore:
    """
    Persistent Vector Store using ChromaDB (or lightweight fallback embedding)
    for candidate and job document chunk index and retrieval.

    Chunking:
      - Resume: project sections preserved as atomic chunks with project_name metadata
      - Job: requirements summary + full description

    Retrieval:
      - Metadata pre-filtering (candidate_id, job_id, section)
      - ChromaDB cosine similarity (primary) or in-memory cosine (fallback)
      - Followed by deterministic intent-aware reranking (entity match + section priority)
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
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning(f"ChromaDB initialization error: {e}")
                self.chroma_client = None

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding via Ollama nomic-embed-text or TF-IDF bag-of-words fallback."""
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

        # TF-IDF / BoW fallback (384 dims)
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
            "project_name": chunk.get("project_name", ""),
            "source": chunk.get("source", ""),
        }

        if self.collection is not None:
            try:
                emb = self._get_embedding(text)
                self.collection.upsert(
                    ids=[chunk_id],
                    embeddings=[emb],
                    documents=[text],
                    metadatas=[meta],
                )
            except Exception as e:
                logger.warning(f"ChromaDB upsert warning for {chunk_id}: {e}")

        # Always maintain in-memory store for fallback
        self.memory_store = [c for c in self.memory_store if c["chunk_id"] != chunk_id]
        self.memory_store.append({
            "chunk_id": chunk_id,
            "text": text,
            "metadata": meta,
            "embedding": self._get_embedding(text),
        })

    def search(
        self,
        query: str,
        candidate_id: Optional[int] = None,
        job_id: Optional[int] = None,
        section_filter: Optional[List[str]] = None,
        top_k: int = 8,
        query_entities: Optional[List[str]] = None,
        intent: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k chunks by metadata pre-filtering + cosine similarity + deterministic reranking.

        Args:
            query:          resolved search query
            candidate_id:   filter results to this candidate (security boundary)
            job_id:         filter results to this job (security boundary)
            section_filter: list of allowed sections (e.g. ["projects"], ["education"])
            top_k:          number of final results after reranking
            query_entities: specific entity tokens extracted from the query for reranking
            intent:         current question intent (drives reranker section bonuses)
        """
        query_emb = self._get_embedding(query)
        results = []
        fetch_k = max(top_k * 2, 10)  # over-retrieve before reranking

        # 1. ChromaDB retrieval
        if self.collection is not None:
            try:
                conditions: List[Dict[str, Any]] = []

                if candidate_id and job_id:
                    conditions.append({"$or": [{"candidate_id": str(candidate_id)}, {"job_id": str(job_id)}]})
                elif candidate_id:
                    conditions.append({"candidate_id": str(candidate_id)})
                elif job_id:
                    conditions.append({"job_id": str(job_id)})

                if section_filter:
                    if len(section_filter) == 1:
                        conditions.append({"section": section_filter[0]})
                    elif len(section_filter) > 1:
                        conditions.append({"$or": [{"section": s} for s in section_filter]})

                where_clause: Dict[str, Any] = {}
                if len(conditions) == 1:
                    where_clause = conditions[0]
                elif len(conditions) > 1:
                    where_clause = {"$and": conditions}

                kwargs: Dict[str, Any] = {
                    "query_embeddings": [query_emb],
                    "n_results": fetch_k,
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
                            "project_name": m.get("project_name", ""),
                            "source": m.get("source", "Document"),
                            "document_type": m.get("document_type", "general"),
                            "similarity_score": round(1.0 - float(dist), 4) if dist <= 1.0 else round(1.0 / (1.0 + float(dist)), 4),
                        })
            except Exception as e:
                logger.warning(f"ChromaDB search query warning: {e}")

        # 2. In-memory fallback if ChromaDB returned nothing
        if not results:
            for item in self.memory_store:
                m = item["metadata"]
                if candidate_id and m.get("candidate_id") and m["candidate_id"] != str(candidate_id):
                    if not (job_id and m.get("job_id") == str(job_id)):
                        continue
                if job_id and m.get("job_id") and m["job_id"] != str(job_id):
                    if not (candidate_id and m.get("candidate_id") == str(candidate_id)):
                        continue
                if section_filter and m.get("section") not in section_filter:
                    continue

                emb = item["embedding"]
                sim = sum(a * b for a, b in zip(query_emb, emb))
                results.append({
                    "text": item["text"],
                    "section": m.get("section", "general"),
                    "project_name": m.get("project_name", ""),
                    "source": m.get("source", "Document"),
                    "document_type": m.get("document_type", "general"),
                    "similarity_score": round(float(sim), 4),
                })
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            results = results[:fetch_k]

        # 3. Deterministic reranking
        if query_entities or intent:
            results = rerank_chunks(results, query_entities or [], intent)
        else:
            results.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

        return results[:top_k]
