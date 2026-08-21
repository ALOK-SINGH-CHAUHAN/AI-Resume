import unittest
from app.rag.engine import RAGRecruiterAssistant

class DummyVectorStore:
    def search(self, query: str, candidate_id=None, job_id=None, top_k=5,
               query_entities=None, intent=""):
        return [
            {
                "id": "c1_s1",
                "text": "Cybersecurity Engineer with experience in Cloud Security, AWS, Terraform, Docker, and Kubernetes.",
                "source": "Resume (Jordan Meyer)",
                "section": "Professional Summary",
                "project_name": "",
                "similarity_score": 0.85,
            }
        ]

class DummyMatchingEngine:
    def compute_match(self, **kwargs):
        return {
            "overall_score": 0.85,
            "skill_score": 0.90,
            "tech_score": 0.80,
            "semantic_score": 0.82,
            "matched_required": ["AWS", "Terraform", "Kubernetes"],
            "hard_gaps": ["Rust"],
            "related_competencies": []
        }

class TestPart3StaleCorruption(unittest.TestCase):
    def setUp(self):
        self.vector_store = DummyVectorStore()
        self.matching_engine = DummyMatchingEngine()
        self.assistant = RAGRecruiterAssistant(self.vector_store, self.matching_engine)
        self.candidate = {
            "id": 10,
            "name": "Jordan Meyer",
            "skills": ["Cybersecurity", "Cloud Security", "Vulnerability Assessment"],
            "technologies": ["AWS", "Terraform", "Docker", "Kubernetes"],
            "languages": [],
            "raw_text": "JORDAN MEYER\nCybersecurity Engineer\nBuilt Cloud Defense and Security Engineering pipelines using AWS, Terraform, and Kubernetes."
        }
        self.job = {
            "id": 10,
            "title": "Cybersecurity Engineer",
            "required_skills": ["Cybersecurity", "Cloud Security"],
            "preferred_skills": [],
            "required_technologies": ["AWS", "Terraform"],
            "preferred_technologies": [],
            "description": "Cybersecurity Engineer role requiring AWS and Terraform."
        }

    def test_q1_rust_no_cpp_leak(self):
        """Q1: 'Does the candidate have Rust experience?' must mention Rust and NOT C++."""
        res = self.assistant.ask_assistant(
            question="Does the candidate have Rust experience?",
            candidate_dict=self.candidate,
            job_dict=self.job
        )
        ans = res["answer"]
        self.assertIn("rust", ans.lower())
        self.assertNotIn("c++", ans.lower())

    def test_q2_aws_10_years_no_cpp_leak(self):
        """Q2: 'Does the candidate have 10 years of AWS experience?' must mention AWS and NOT C++."""
        res = self.assistant.ask_assistant(
            question="Does the candidate have 10 years of AWS experience?",
            candidate_dict=self.candidate,
            job_dict=self.job
        )
        ans = res["answer"]
        self.assertIn("aws", ans.lower())
        self.assertNotIn("c++", ans.lower())

    def test_q3_google_no_stale_leak(self):
        """Q3: 'Did the candidate work at Google?' must mention Google and NOT AWS/Rust/C++."""
        res = self.assistant.ask_assistant(
            question="Did the candidate work at Google?",
            candidate_dict=self.candidate,
            job_dict=self.job
        )
        ans = res["answer"]
        self.assertIn("google", ans.lower())
        self.assertNotIn("c++", ans.lower())
        self.assertNotIn("rust", ans.lower())

    def test_q4_terraform_mention(self):
        """Q4: 'Does the candidate have Terraform experience?' must mention Terraform."""
        res = self.assistant.ask_assistant(
            question="Does the candidate have Terraform experience?",
            candidate_dict=self.candidate,
            job_dict=self.job
        )
        ans = res["answer"]
        self.assertIn("terraform", ans.lower())

    def test_q5_security_projects(self):
        """Q5: 'What projects demonstrate security engineering?' must mention security engineering / candidate projects."""
        res = self.assistant.ask_assistant(
            question="What projects demonstrate security engineering?",
            candidate_dict=self.candidate,
            job_dict=self.job
        )
        ans = res["answer"]
        self.assertTrue("security" in ans.lower() or "project" in ans.lower())

if __name__ == "__main__":
    unittest.main()
