import os
import sys

# Add backend directory to sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from app.nlp.extractor import NLPExtractor
from app.matching.engine import MatchingEngine

data_dir = os.path.join(backend_dir, "app", "data")
extractor = NLPExtractor(data_dir)
matching_engine = MatchingEngine(data_dir, extractor)

def test_exact_sample_sentence():
    sample_text = "I worked in the AI/ML Department and worked with CNN Models using Python"
    result = extractor.extract(sample_text)
    
    print("\n--- Test 1: Assignment Sample Sentence Extraction ---")
    print("Input:", sample_text)
    print("Output:", result)

    assert "Machine Learning" in result["skill"], "Expected 'Machine Learning' in skill"
    assert "CNN" in result["technology"], "Expected 'CNN' in technology"
    assert "Python" in result["language"], "Expected 'Python' in language"
    print("PASS: Sample sentence extracted correctly!")

def test_synonyms_and_dedup():
    text = "I use py, python3, tf, tensorflow, reactjs, and ML"
    result = extractor.extract(text)
    
    print("\n--- Test 2: Synonyms & Deduplication ---")
    print("Input:", text)
    print("Output:", result)

    assert result["language"].count("Python") == 1
    assert result["technology"].count("TensorFlow") == 1
    assert result["technology"].count("React") == 1
    assert result["skill"].count("Machine Learning") == 1
    print("PASS: Synonyms resolved and deduplicated!")

def test_matching_engine():
    cand_skills = ["Machine Learning", "Deep Learning"]
    cand_tech = ["CNN", "TensorFlow"]
    cand_lang = ["Python"]
    cand_text = "Experienced AI Engineer working with Machine Learning, Deep Learning, CNN, TensorFlow, and Python."

    jd_text = "We are seeking a Machine Learning Engineer with required skills in Machine Learning and Python, and preferred tech in CNN."
    reqs = matching_engine.extract_jd_requirements(jd_text)

    match_res = matching_engine.compute_match(
        candidate_skills=cand_skills,
        candidate_tech=cand_tech,
        candidate_lang=cand_lang,
        candidate_text=cand_text,
        jd_req_skills=reqs["required_skills"],
        jd_pref_skills=reqs["preferred_skills"],
        jd_req_tech=reqs["required_technologies"],
        jd_pref_tech=reqs["preferred_technologies"],
        jd_text=jd_text
    )

    print("\n--- Test 3: Matching Engine ---")
    print("Match Results:", match_res)
    assert match_res["skill_score"] >= 0.8, "Expected high skill score for matching skills"
    assert "Machine Learning" in match_res["matched_skills"] or "CNN" in match_res["matched_skills"]
    print("PASS: Matching engine computed valid scores!")

if __name__ == "__main__":
    test_exact_sample_sentence()
    test_synonyms_and_dedup()
    test_matching_engine()
    print("\nALL BACKEND UNIT TESTS PASSED SUCCESSFULLY!")
