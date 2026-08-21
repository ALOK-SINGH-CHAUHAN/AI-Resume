import os
import sys
import json
import uuid
import re
import html
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import SQLModel, create_engine, Session, select

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import pymupdf as fitz  # PyMuPDF

from app.nlp.extractor import NLPExtractor
from app.matching.engine import MatchingEngine
from app.models.schemas import Candidate, Job, Match

DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, "sqlite.db")
sqlite_url = f"sqlite:///{DB_PATH}"
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

app = FastAPI(title="AI Recruiter API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = NLPExtractor(DATA_DIR)
matching_engine = MatchingEngine(DATA_DIR, extractor)

# Part 3 — RAG Vector Store & Recruiter Assistant Initialization
from app.rag.store import RAGVectorStore
from app.rag.engine import RAGRecruiterAssistant

CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
rag_vector_store = RAGVectorStore(CHROMA_DIR)
rag_assistant = RAGRecruiterAssistant(rag_vector_store, matching_engine)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    with Session(engine) as session:
        existing_jobs = session.exec(select(Job)).all()
        titles = {j.title for j in existing_jobs}

        # Seed realistic distinct job descriptions if missing
        seed_jobs = [
            {
                "title": "RAG & LLM Systems Engineer",
                "description": "We are seeking a RAG & LLM Systems Engineer with required skills in RAG, Retrieval Augmented Generation, Embeddings, Vector Search, and Python. Required technologies: LangChain, ChromaDB, and PyTorch. Preferred skills: Semantic Search, Information Retrieval, and Natural Language Processing. Bonus: Pinecone and Docker experience."
            },
            {
                "title": "Senior Full Stack Developer",
                "description": "Looking for a Senior Full Stack Developer with required skills in Full Stack Development, Frontend Development, Backend Development, and REST APIs. Required technologies: React, Node.js, and PostgreSQL. Preferred skills: Database Management and System Design. Bonus: Next.js, Docker, and TypeScript."
            },
            {
                "title": "AI Platform Architect",
                "description": "Hiring an AI Platform Architect with required skills in Platform Engineering, MLOps, Cloud Computing, and Machine Learning. Required technologies: Kubernetes, Docker, and AWS. Preferred skills: System Architecture and Distributed Systems. Bonus: Terraform, Airflow, and Go."
            }
        ]

        for sj in seed_jobs:
            if sj["title"] not in titles:
                extracted_reqs = matching_engine.extract_jd_requirements(f"{sj['title']}\n{sj['description']}")
                db_job = Job(
                    title=sj["title"],
                    description=sj["description"],
                    required_skills_json=json.dumps(extracted_reqs["required_skills"]),
                    preferred_skills_json=json.dumps(extracted_reqs["preferred_skills"]),
                    required_technologies_json=json.dumps(extracted_reqs["required_technologies"]),
                    preferred_technologies_json=json.dumps(extracted_reqs["preferred_technologies"]),
                    languages_json=json.dumps(extracted_reqs["languages"])
                )
                session.add(db_job)
        session.commit()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ai-recruiter",
        "version": "1.0"
    }

# --- PART 1: EXTRACTION ENDPOINTS ---

class ExtractRequest(BaseModel):
    text: str

@app.post("/extract")
def extract_entities(req: ExtractRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Please provide non-empty text to analyze.")
    result = extractor.extract(req.text)
    return result

class CandidateCreate(BaseModel):
    name: str
    contact_info: Optional[str] = None
    raw_text: str
    skills: List[str] = []
    technologies: List[str] = []
    languages: List[str] = []
    resume_file_path: Optional[str] = None

import hashlib

def compute_text_hash(text: str) -> str:
    """Compute sha256 hash of normalized text for exact duplicate detection."""
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def normalize_display_name(name: str) -> str:
    """Title-case candidate name while preserving uppercase initials (e.g. 'alok' -> 'Alok')."""
    if not name or not name.strip():
        return "Unknown Candidate"
    clean = " ".join(name.strip().split())
    # If all uppercase or all lowercase, title-case it
    if clean.isupper() or clean.islower():
        return clean.title()
    return clean

@app.post("/candidate")
def create_candidate(cand: CandidateCreate, session: Session = Depends(get_session)):
    norm_name = normalize_display_name(cand.name)
    t_hash = compute_text_hash(cand.raw_text)

    # 1. Check exact resume text fingerprint match
    existing_by_hash = session.exec(
        select(Candidate).where(Candidate.text_hash == t_hash)
    ).first()

    if existing_by_hash:
        return {
            "id": existing_by_hash.id,
            "name": existing_by_hash.name,
            "contact_info": existing_by_hash.contact_info,
            "raw_text": existing_by_hash.raw_text,
            "skills": existing_by_hash.skills,
            "technologies": existing_by_hash.technologies,
            "languages": existing_by_hash.languages,
            "created_at": existing_by_hash.created_at,
            "is_duplicate": True,
            "message": f"Candidate profile already exists as ID #{existing_by_hash.id} ({existing_by_hash.name}). Duplicate record was not created."
        }

    # 2. Check normalized name + contact info match (if contact info provided)
    if cand.contact_info and cand.contact_info.strip():
        norm_contact = cand.contact_info.strip().lower()
        all_cands = session.exec(select(Candidate)).all()
        for existing in all_cands:
            if existing.contact_info and existing.contact_info.strip().lower() == norm_contact:
                if existing.name.strip().lower() == norm_name.lower():
                    return {
                        "id": existing.id,
                        "name": existing.name,
                        "contact_info": existing.contact_info,
                        "raw_text": existing.raw_text,
                        "skills": existing.skills,
                        "technologies": existing.technologies,
                        "languages": existing.languages,
                        "created_at": existing.created_at,
                        "is_duplicate": True,
                        "message": f"Candidate profile already exists as ID #{existing.id} ({existing.name}). Duplicate record was not created."
                    }

    db_cand = Candidate(
        name=norm_name,
        contact_info=cand.contact_info,
        raw_text=cand.raw_text,
        resume_file_path=cand.resume_file_path,
        text_hash=t_hash,
        skills_json=json.dumps(cand.skills),
        technologies_json=json.dumps(cand.technologies),
        languages_json=json.dumps(cand.languages)
    )
    session.add(db_cand)
    session.commit()
    session.refresh(db_cand)
    return {
        "id": db_cand.id,
        "name": db_cand.name,
        "contact_info": db_cand.contact_info,
        "raw_text": db_cand.raw_text,
        "skills": db_cand.skills,
        "technologies": db_cand.technologies,
        "languages": db_cand.languages,
        "created_at": db_cand.created_at,
        "is_duplicate": False,
        "message": "Candidate profile saved successfully."
    }

@app.get("/candidates")
def list_candidates(session: Session = Depends(get_session)):
    candidates = session.exec(select(Candidate)).all()
    res = []
    for c in candidates:
        res.append({
            "id": c.id,
            "name": c.name,
            "contact_info": c.contact_info,
            "raw_text": c.raw_text,
            "skills": c.skills,
            "technologies": c.technologies,
            "languages": c.languages,
            "created_at": c.created_at
        })
    return res

@app.delete("/candidate/{candidate_id}")
def delete_candidate(candidate_id: int, session: Session = Depends(get_session)):
    cand = session.get(Candidate, candidate_id)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    session.delete(cand)
    session.commit()
    return {"message": f"Candidate #{candidate_id} ({cand.name}) deleted successfully."}

@app.get("/candidate/{candidate_id}")
def get_candidate(candidate_id: int, session: Session = Depends(get_session)):
    c = session.get(Candidate, candidate_id)
    if not c:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return {
        "id": c.id,
        "name": c.name,
        "contact_info": c.contact_info,
        "raw_text": c.raw_text,
        "skills": c.skills,
        "technologies": c.technologies,
        "languages": c.languages,
        "created_at": c.created_at
    }

@app.delete("/candidate/{candidate_id}")
def delete_candidate(candidate_id: int, session: Session = Depends(get_session)):
    c = session.get(Candidate, candidate_id)
    if not c:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    session.delete(c)
    session.commit()
    return {"message": "Candidate deleted successfully."}

# --- PART 2: MATCHING & JOB ENDPOINTS ---

class JobCreate(BaseModel):
    title: str
    description: str

@app.post("/job")
def create_job(job_in: JobCreate, session: Session = Depends(get_session)):
    if not job_in.description.strip():
        raise HTTPException(status_code=400, detail="Could not extract requirements from this job description — please add more detail.")

    extracted_reqs = matching_engine.extract_jd_requirements(job_in.description)

    db_job = Job(
        title=job_in.title,
        description=job_in.description,
        required_skills_json=json.dumps(extracted_reqs["required_skills"]),
        preferred_skills_json=json.dumps(extracted_reqs["preferred_skills"]),
        required_technologies_json=json.dumps(extracted_reqs["required_technologies"]),
        preferred_technologies_json=json.dumps(extracted_reqs["preferred_technologies"]),
        languages_json=json.dumps(extracted_reqs["languages"])
    )
    session.add(db_job)
    session.commit()
    session.refresh(db_job)

    return {
        "id": db_job.id,
        "title": db_job.title,
        "description": db_job.description,
        "required_skills": db_job.required_skills,
        "preferred_skills": db_job.preferred_skills,
        "required_technologies": db_job.required_technologies,
        "preferred_technologies": db_job.preferred_technologies,
        "created_at": db_job.created_at
    }

@app.get("/jobs")
def list_jobs(session: Session = Depends(get_session)):
    jobs = session.exec(select(Job)).all()
    res = []
    for j in jobs:
        res.append({
            "id": j.id,
            "title": j.title,
            "description": j.description,
            "required_skills": j.required_skills,
            "preferred_skills": j.preferred_skills,
            "required_technologies": j.required_technologies,
            "preferred_technologies": j.preferred_technologies,
            "created_at": j.created_at
        })
    return res

@app.get("/job/{job_id}")
def get_job(job_id: int, session: Session = Depends(get_session)):
    j = session.get(Job, job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "id": j.id,
        "title": j.title,
        "description": j.description,
        "required_skills": j.required_skills,
        "preferred_skills": j.preferred_skills,
        "required_technologies": j.required_technologies,
        "preferred_technologies": j.preferred_technologies,
        "created_at": j.created_at
    }

class MatchRequest(BaseModel):
    candidate_id: int
    job_id: int

@app.post("/match")
def match_candidate_job(req: MatchRequest, session: Session = Depends(get_session)):
    cand = session.get(Candidate, req.candidate_id)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    job = session.get(Job, req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    res = matching_engine.compute_match(
        candidate_skills=cand.skills,
        candidate_tech=cand.technologies,
        candidate_lang=cand.languages,
        candidate_text=cand.raw_text,
        jd_req_skills=job.required_skills,
        jd_pref_skills=job.preferred_skills,
        jd_req_tech=job.required_technologies,
        jd_pref_tech=job.preferred_technologies,
        jd_text=job.description
    )

    db_match = Match(
        candidate_id=cand.id,
        job_id=job.id,
        skill_score=res["skill_score"],
        tech_score=res["tech_score"],
        semantic_score=res["semantic_score"],
        overall_score=res["overall_score"],
        matched_skills_json=json.dumps(res["matched_skills"]),
        missing_skills_json=json.dumps(res["missing_skills"])
    )
    session.add(db_match)
    session.commit()

    return {
        "candidate_id": cand.id,
        "candidate_name": cand.name,
        "job_id": job.id,
        "job_title": job.title,
        "skill_score": res["skill_score"],
        "tech_score": res["tech_score"],
        "semantic_score": res["semantic_score"],
        "overall_score": res["overall_score"],
        "weighted_contributions": res.get("weighted_contributions", {}),
        "matched_skills": res["matched_skills"],
        "matched_required": res.get("matched_required", []),
        "matched_preferred": res.get("matched_preferred", []),
        "matched_bonus": res.get("matched_bonus", []),
        "missing_skills": res["missing_skills"],
        "missing_required": res.get("missing_required", []),
        "missing_preferred": res.get("missing_preferred", []),
        "related_competencies": res.get("related_competencies", []),
        "extra_skills": res.get("extra_skills", []),
        "hard_gaps": res.get("hard_gaps", []),
        "has_hard_gaps": res.get("has_hard_gaps", False),
        "evidence": res.get("evidence", []),
        "score_reasons": res.get("score_reasons", [])
    }

class RoleRecRequest(BaseModel):
    candidate_id: int
    top_n: Optional[int] = 5

@app.post("/recommend-roles")
def recommend_roles(req: RoleRecRequest, session: Session = Depends(get_session)):
    cand = session.get(Candidate, req.candidate_id)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    roles = matching_engine.recommend_roles(
        candidate_skills=cand.skills,
        candidate_tech=cand.technologies,
        candidate_lang=cand.languages,
        top_n=req.top_n or 5
    )
    return {"candidate_id": cand.id, "roles": roles}

@app.get("/matches/{job_id}")
def rank_candidates_for_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    candidates = session.exec(select(Candidate)).all()
    rankings = []
    for cand in candidates:
        match_res = matching_engine.compute_match(
            candidate_skills=cand.skills,
            candidate_tech=cand.technologies,
            candidate_lang=cand.languages,
            candidate_text=cand.raw_text,
            jd_req_skills=job.required_skills,
            jd_pref_skills=job.preferred_skills,
            jd_req_tech=job.required_technologies,
            jd_pref_tech=job.preferred_technologies,
            jd_text=job.description
        )
        rankings.append({
            "candidate_id": cand.id,
            "candidate_name": cand.name,
            "overall_score": match_res["overall_score"],
            "skill_score": match_res["skill_score"],
            "tech_score": match_res["tech_score"],
            "semantic_score": match_res["semantic_score"],
            "matched_skills": match_res["matched_skills"],
            "matched_required": match_res.get("matched_required", []),
            "matched_preferred": match_res.get("matched_preferred", []),
            "missing_skills": match_res["missing_skills"],
            "related_competencies": match_res.get("related_competencies", []),
            "extra_skills": match_res.get("extra_skills", []),
            "hard_gaps": match_res.get("hard_gaps", []),
            "has_hard_gaps": match_res.get("has_hard_gaps", False),
        })

    rankings.sort(key=lambda x: (x["overall_score"], x["skill_score"], x["tech_score"]), reverse=True)
    return {"job_id": job.id, "job_title": job.title, "rankings": rankings}

# --- REMOTIVE REAL JOB DISCOVERY ENDPOINT ---

@app.get("/remotive/recommended/{candidate_id}")
def get_remotive_recommended_jobs(candidate_id: int, session: Session = Depends(get_session)):
    cand = session.get(Candidate, candidate_id)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    # 1. Generate effective keyword search terms from candidate's extracted profile & recommended roles
    top_roles = matching_engine.recommend_roles(
        candidate_skills=cand.skills,
        candidate_tech=cand.technologies,
        candidate_lang=cand.languages,
        top_n=3
    )

    # Extract individual search keywords from roles and candidate skills
    keyword_pool = []
    for r in top_roles:
        for word in r["role_name"].split():
            if len(word) > 2 and word.lower() not in ("engineer", "developer", "architect", "senior", "lead"):
                keyword_pool.append(word.lower())

    for s in cand.skills + cand.technologies + cand.languages:
        clean_s = s.lower()
        if clean_s in ("ai", "rag", "ml", "python", "pytorch", "docker", "aws", "kubernetes", "golang", "go", "faiss", "milvus", "redis"):
            keyword_pool.append(clean_s)
        elif len(clean_s) > 2:
            keyword_pool.append(clean_s)

    # Deduplicate while preserving priority
    search_terms = list(dict.fromkeys(keyword_pool))
    if not search_terms:
        search_terms = ["python", "ai", "backend"]

    import urllib.request
    import html

    raw_jobs = []
    seen_ids = set()

    # Query Remotive API for top 5 candidate keywords
    for term in search_terms[:5]:
        try:
            url = f"https://remotive.com/api/remote-jobs?search={urllib.parse.quote(term)}"
            req = urllib.request.Request(url, headers={"User-Agent": "AI-Recruiter-Engine/1.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                for item in data.get("jobs", []):
                    job_id = item.get("id")
                    if job_id not in seen_ids:
                        seen_ids.add(job_id)
                        raw_jobs.append(item)
        except Exception as e:
            print(f"Warning fetching Remotive for term {term}: {e}")

    # Fallback search if no jobs returned
    if not raw_jobs:
        try:
            url = "https://remotive.com/api/remote-jobs?search=python"
            req = urllib.request.Request(url, headers={"User-Agent": "AI-Recruiter-Engine/1.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                raw_jobs = data.get("jobs", [])[:15]
        except Exception:
            pass

    results = []
    for item in raw_jobs:
        title = item.get("title", "Remote Role")
        company = item.get("company_name", "Remotive Partner")
        job_url = item.get("url", "https://remotive.com")
        category = item.get("category", "")
        job_type = item.get("job_type", "Full-time")
        candidate_location = item.get("candidate_required_location", "Worldwide")
        publication_date = item.get("publication_date", "")[:10]
        
        # Clean HTML description
        raw_desc = item.get("description", "")
        clean_desc = re.sub(r"<[^>]+>", " ", raw_desc)
        clean_desc = html.unescape(clean_desc)
        clean_desc = " ".join(clean_desc.split())[:1500]

        # Extract requirements from clean_desc
        extracted_reqs = matching_engine.extract_jd_requirements(f"{title}\n{clean_desc}")

        match_res = matching_engine.compute_match(
            candidate_skills=cand.skills,
            candidate_tech=cand.technologies,
            candidate_lang=cand.languages,
            candidate_text=cand.raw_text,
            jd_req_skills=extracted_reqs["required_skills"],
            jd_pref_skills=extracted_reqs["preferred_skills"],
            jd_req_tech=extracted_reqs["required_technologies"],
            jd_pref_tech=extracted_reqs["preferred_technologies"],
            jd_text=f"{title}\n{clean_desc}"
        )

        # Include all candidate-matched jobs with overlap or score >= 0.20
        has_overlap = (
            len(match_res["matched_skills"]) > 0 or
            len(match_res.get("matched_required", [])) > 0 or
            len(match_res.get("matched_preferred", [])) > 0
        )
        if has_overlap or match_res["overall_score"] >= 0.20:
            results.append({
                "remotive_id": item.get("id"),
                "title": title,
                "company_name": company,
                "url": job_url,  # Official Remotive listing URL
                "category": category,
                "job_type": job_type,
                "location": candidate_location,
                "publication_date": publication_date,
                "source": "Remotive",
                "description_snippet": clean_desc[:250] + "..." if len(clean_desc) > 250 else clean_desc,
                "overall_score": match_res["overall_score"],
                "skill_score": match_res["skill_score"],
                "tech_score": match_res["tech_score"],
                "semantic_score": match_res["semantic_score"],
                "matched_skills": match_res["matched_skills"],
                "matched_required": match_res.get("matched_required", []),
                "matched_preferred": match_res.get("matched_preferred", []),
                "missing_skills": match_res["missing_skills"],
                "hard_gaps": match_res.get("hard_gaps", []),
                "has_hard_gaps": match_res.get("has_hard_gaps", False),
                "score_reasons": match_res.get("score_reasons", [])
            })

    results.sort(key=lambda x: (x["overall_score"], x["skill_score"]), reverse=True)
    return {
        "candidate_id": cand.id,
        "candidate_name": cand.name,
        "total_found": len(results),
        "recommended_jobs": results[:10]
    }

# --- PDF UPLOAD ENDPOINT ---

@app.post("/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    """
    Extract raw text from PDF/TXT file only.
    NLP entity extraction is NOT performed here — the caller must
    explicitly invoke POST /extract after review.
    """
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and text files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size exceeds 5MB limit.")

    extracted_text = ""
    if file.filename.lower().endswith(".pdf"):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                extracted_text += page.get_text() + "\n"
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read this PDF — try re-exporting or pasting text instead.")
    else:
        extracted_text = file_bytes.decode("utf-8", errors="ignore")

    if not extracted_text.strip():
        raise HTTPException(status_code=422, detail="No readable text found in file.")

    # Return text only — NLP extraction happens separately via POST /extract
    char_count = len(extracted_text.strip())
    preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
    return {
        "filename": file.filename,
        "extracted_text": extracted_text,
        "preview": preview,
        "char_count": char_count,
    }

# --- PART 3: RAG RECRUITER INTELLIGENCE ASSISTANT ENDPOINTS ---

class AssistantChatRequest(BaseModel):
    candidate_id: int
    job_id: Optional[int] = None
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None

@app.post("/assistant/chat")
def assistant_chat(req: AssistantChatRequest, session: Session = Depends(get_session)):
    cand = session.get(Candidate, req.candidate_id)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    cand_dict = {
        "id": cand.id,
        "name": cand.name,
        "skills": cand.skills,
        "technologies": cand.technologies,
        "languages": cand.languages,
        "raw_text": cand.raw_text
    }

    # Index candidate in ChromaDB vector store
    rag_vector_store.index_candidate(cand_dict)

    job_dict = None
    if req.job_id:
        job = session.get(Job, req.job_id)
        if job:
            job_dict = {
                "id": job.id,
                "title": job.title,
                "description": job.description,
                "required_skills": job.required_skills,
                "required_technologies": job.required_technologies,
                "preferred_skills": job.preferred_skills,
                "preferred_technologies": job.preferred_technologies
            }
            # Index job in ChromaDB vector store
            rag_vector_store.index_job(job_dict)

    res = rag_assistant.ask_assistant(
        question=req.question,
        candidate_dict=cand_dict,
        job_dict=job_dict,
        chat_history=req.chat_history
    )
    return res

@app.get("/assistant/debug-retrieval")
def debug_retrieval(
    candidate_id: int,
    question: str = "RAG experience and background",
    job_id: Optional[int] = None,
    session: Session = Depends(get_session)
):
    cand = session.get(Candidate, candidate_id)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    cand_dict = {
        "id": cand.id,
        "name": cand.name,
        "skills": cand.skills,
        "technologies": cand.technologies,
        "languages": cand.languages,
        "raw_text": cand.raw_text
    }
    rag_vector_store.index_candidate(cand_dict)

    if job_id:
        job = session.get(Job, job_id)
        if job:
            job_dict = {
                "id": job.id,
                "title": job.title,
                "description": job.description,
                "required_skills": job.required_skills,
                "required_technologies": job.required_technologies,
                "preferred_skills": job.preferred_skills,
                "preferred_technologies": job.preferred_technologies
            }
            rag_vector_store.index_job(job_dict)

    chunks = rag_vector_store.search(query=question, candidate_id=candidate_id, job_id=job_id, top_k=5)
    return {
        "candidate_id": candidate_id,
        "job_id": job_id,
        "query": question,
        "retrieved_chunks_count": len(chunks),
        "retrieved_chunks": chunks
    }
