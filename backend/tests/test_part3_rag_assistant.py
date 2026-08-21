import os
import sys
import unittest
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.nlp.extractor import NLPExtractor
from app.matching.engine import MatchingEngine
from app.rag.store import RAGVectorStore, semantic_chunk_resume, semantic_chunk_job
from app.rag.engine import RAGRecruiterAssistant

class TestPart3RAGAssistant(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(BASE_DIR, "app", "data")
        self.chroma_dir = os.path.join(self.data_dir, "test_chroma_db")
        self.extractor = NLPExtractor(self.data_dir)
        self.engine = MatchingEngine(self.data_dir, self.extractor)
        self.store = RAGVectorStore(self.chroma_dir)
        self.assistant = RAGRecruiterAssistant(self.store, self.engine)

        self.candidate = {
            "id": 3,
            "name": "Alok Singh",
            "skills": ["RAG", "Distributed Systems", "Vector Search", "AI Agents"],
            "technologies": ["LangGraph", "ChromaDB", "PyTorch", "Docker", "Faiss", "Python"],
            "languages": ["Python", "Go"],
            "raw_text": "Experienced AI Systems Engineer building production RAG pipelines with LangGraph, ChromaDB, PyTorch, Docker, and FAISS."
        }

        self.job = {
            "id": 2,
            "title": "RAG & LLM Systems Engineer",
            "description": "We are seeking a RAG & LLM Systems Engineer with required skills in RAG, Vector Search, and LangChain.",
            "required_skills": ["RAG", "Vector Search"],
            "required_technologies": ["LangChain", "ChromaDB", "PyTorch"],
            "preferred_skills": ["AI Agents"],
            "preferred_technologies": ["Docker"]
        }

    def test_01_semantic_section_chunking(self):
        """Test semantic section chunking for resumes and jobs."""
        cand_chunks = semantic_chunk_resume(self.candidate)
        self.assertTrue(len(cand_chunks) >= 2)
        self.assertEqual(cand_chunks[0]["document_type"], "resume")
        self.assertEqual(cand_chunks[0]["section"], "skills_summary")

        job_chunks = semantic_chunk_job(self.job)
        self.assertTrue(len(job_chunks) >= 2)
        self.assertEqual(job_chunks[0]["document_type"], "job")
        self.assertEqual(job_chunks[0]["section"], "requirements")

    def test_02_rag_vector_store_indexing_and_search(self):
        """Test document indexing and ChromaDB vector search retrieval."""
        self.store.index_candidate(self.candidate)
        self.store.index_job(self.job)

        results = self.store.search(
            query="RAG pipeline and vector search",
            candidate_id=3,
            job_id=2,
            top_k=4
        )

        self.assertTrue(len(results) > 0)
        sections = [r["section"] for r in results]
        self.assertTrue(any(s in ("skills_summary", "background_and_experience", "requirements", "full_description") for s in sections))

    def test_03_recruiter_assistant_grounding_and_citations(self):
        """Test RAGRecruiterAssistant grounded prompt assembly and citations."""
        self.store.index_candidate(self.candidate)
        self.store.index_job(self.job)

        res = self.assistant.ask_assistant(
            question="Why is Alok a good fit for this job?",
            candidate_dict=self.candidate,
            job_dict=self.job
        )

        self.assertIsNotNone(res["answer"])
        self.assertIn("Alok", res["answer"])
        self.assertTrue(len(res["evidence_citations"]) > 0)

        # Deterministic match scores must be present and preserved
        self.assertIsNotNone(res["deterministic_match"])
        self.assertIn("overall_score", res["deterministic_match"])
        self.assertTrue(res["deterministic_match"]["overall_score"] > 0)

if __name__ == "__main__":
    unittest.main()
