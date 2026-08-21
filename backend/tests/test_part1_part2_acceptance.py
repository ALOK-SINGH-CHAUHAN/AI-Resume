import os
import sys
import unittest
import json
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.nlp.extractor import NLPExtractor
from app.matching.engine import MatchingEngine

class TestPart1Part2Acceptance(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(BASE_DIR, "app", "data")
        self.extractor = NLPExtractor(self.data_dir)
        self.engine = MatchingEngine(self.data_dir, self.extractor)

    def test_01_candidate_name_and_fingerprint_normalization(self):
        """Test name title-casing and text_hash duplicate fingerprinting."""
        from app.main import normalize_display_name, compute_text_hash

        self.assertEqual(normalize_display_name("alok"), "Alok")
        self.assertEqual(normalize_display_name("ALOK"), "Alok")
        self.assertEqual(normalize_display_name("priya sharma"), "Priya Sharma")
        self.assertEqual(normalize_display_name("Alok Singh"), "Alok Singh")

        text1 = "Experienced AI Systems Engineer working with RAG, Vector Search, and Python."
        text2 = "  experienced  ai  systems engineer   working with rag, vector search, and python. "
        
        hash1 = compute_text_hash(text1)
        hash2 = compute_text_hash(text2)
        self.assertEqual(hash1, hash2)

    def test_02_nlp_extraction_provenance_and_negation(self):
        """Test entity extraction with authentic provenance snippets and negation filtering."""
        text = "I never used React. Currently learning LangChain and building RAG pipelines with ChromaDB. Previously worked professionally with PyTorch and FastAPI at a startup."
        res = self.extractor.extract(text)

        # React should be negated and excluded from skill/tech lists
        self.assertNotIn("React", res["technology"])
        self.assertIn("React", [e["canonical"] for e in res.get("negated_entities", [])])

        # Extracted items
        self.assertIn("RAG", res["skill"])
        self.assertIn("LangChain", res["technology"])
        self.assertIn("ChromaDB", res["technology"])
        self.assertIn("PyTorch", res["technology"])
        self.assertIn("FastAPI", res["technology"])

        # Context classification
        evidence_by_canonical = {e["canonical"]: e for e in res["evidence"]}
        self.assertEqual(evidence_by_canonical["LangChain"]["experience_context"], "learning")
        self.assertEqual(evidence_by_canonical["PyTorch"]["experience_context"], "professional")

        # Provenance sentence must be authentic excerpt from text
        self.assertIn("LangChain and building RAG", evidence_by_canonical["LangChain"]["source_sentence"])

    def test_03_matching_engine_partial_credit_and_hard_gaps(self):
        """Test direct match (1.0x), related match partial credit (0.5x), and hard gap detection."""
        cand_skills = ["RAG", "Distributed Systems", "AI Agents"]
        cand_tech = ["LangGraph", "ChromaDB", "PyTorch", "Docker"]
        cand_lang = ["Python"]
        cand_text = "Built RAG systems using LangGraph, ChromaDB, PyTorch, and Docker."

        # Job requires LangChain (candidate has LangGraph — related ecosystem technology)
        jd_req_skills = ["RAG"]
        jd_pref_skills = []
        jd_req_tech = ["LangChain", "PyTorch"]
        jd_pref_tech = ["Docker"]
        jd_text = "We need a RAG Engineer with LangChain, PyTorch, and Docker experience."

        res = self.engine.compute_match(
            candidate_skills=cand_skills,
            candidate_tech=cand_tech,
            candidate_lang=cand_lang,
            candidate_text=cand_text,
            jd_req_skills=jd_req_skills,
            jd_pref_skills=jd_pref_skills,
            jd_req_tech=jd_req_tech,
            jd_pref_tech=jd_pref_tech,
            jd_text=jd_text,
        )

        # PyTorch = direct req match, LangChain = related match via LangGraph
        self.assertIn("PyTorch", res["matched_required"])
        self.assertIn("LangChain", res["matched_required_related"])
        self.assertFalse(res["has_hard_gaps"])
        self.assertEqual(len(res["hard_gaps"]), 0)

        # Weighted contributions
        self.assertEqual(res["weighted_contributions"]["weights"]["skill"], 0.45)
        self.assertEqual(res["weighted_contributions"]["weights"]["tech"], 0.30)
        self.assertEqual(res["weighted_contributions"]["weights"]["semantic"], 0.25)

    def test_04_role_recommendation_threshold(self):
        """Test minimum evidence threshold filters out nonsensical roles (e.g. Security Engineer for AI candidate)."""
        cand_skills = ["RAG", "Distributed Systems", "Vector Search", "AI Agents", "Generative AI"]
        cand_tech = ["PyTorch", "Docker", "Kubernetes", "Milvus", "Redis", "FAISS", "LangGraph", "AWS"]
        cand_lang = ["Python", "Go"]

        recs = self.engine.recommend_roles(
            candidate_skills=cand_skills,
            candidate_tech=cand_tech,
            candidate_lang=cand_lang,
            top_n=5
        )

        role_names = [r["role_name"] for r in recs]
        
        # Security Engineer must NOT be recommended
        self.assertNotIn("Security Engineer", role_names)
        self.assertNotIn("Full Stack Developer", role_names)

        # Highly relevant niche AI/ML roles MUST be recommended
        self.assertIn("AI Agent Engineer", role_names)

if __name__ == "__main__":
    unittest.main()
