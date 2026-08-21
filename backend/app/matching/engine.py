import json
import os
import re
from typing import Dict, List, Any, Set, Tuple, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except ImportError:
    HAVE_ST = False


class MatchingEngine:
    def __init__(self, data_dir: str, nlp_extractor):
        self.data_dir = data_dir
        self.extractor = nlp_extractor
        self.model = None

        roles_path = os.path.join(self.data_dir, "job_roles.json")
        if os.path.exists(roles_path):
            with open(roles_path, "r", encoding="utf-8") as f:
                self.job_roles: List[Dict[str, Any]] = json.load(f)
        else:
            self.job_roles = []

        ontology_path = os.path.join(self.data_dir, "skill_ontology.json")
        if os.path.exists(ontology_path):
            with open(ontology_path, "r", encoding="utf-8") as f:
                self.skill_ontology: Dict[str, Any] = json.load(f)
        else:
            self.skill_ontology = {}

        self._init_semantic_model()

    def _init_semantic_model(self):
        if HAVE_ST:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: Failed to load SentenceTransformer model: {e}")
                self.model = None

    # -------------------------------------------------------------------
    # Ontology: related competency detection
    # -------------------------------------------------------------------

    def _get_related_skills(self, skill_name: str) -> Set[str]:
        """Return parent + children of a skill from the ontology (one hop)."""
        related: Set[str] = set()
        node = self.skill_ontology.get(skill_name)
        if node:
            if node.get("parent"):
                related.add(node["parent"])
            related.update(node.get("children", []))
        return related

    def _classify_skill_match(
        self, term: str, candidate_set: Set[str]
    ) -> str:
        """
        Classify a job requirement term against candidate skills:
        - 'direct': exact match in candidate set
        - 'related': candidate has a parent/child in ontology
        - 'missing': no match
        """
        if term.lower() in candidate_set:
            return "direct"
        related = self._get_related_skills(term)
        for rel in related:
            if rel.lower() in candidate_set:
                return "related"
        return "missing"

    # -------------------------------------------------------------------
    # JD Requirement Extraction: required / preferred / bonus
    # -------------------------------------------------------------------

    REQUIRED_CUES = [
        "required", "must have", "must", "mandatory", "essential",
        "need", "needs", "necessary", "you will need",
    ]
    PREFERRED_CUES = [
        "preferred", "nice to have", "plus", "optional", "desired",
        "ideal", "bonus if", "beneficial",
    ]
    BONUS_CUES = [
        "bonus", "advantage", "advantageous", "great to have",
        "nice if", "would be great",
    ]

    def extract_jd_requirements(self, jd_text: str) -> Dict[str, List[str]]:
        raw_extraction = self.extractor.extract(jd_text)
        all_skills = raw_extraction.get("skill", [])
        all_tech = raw_extraction.get("technology", [])
        all_lang = raw_extraction.get("language", [])

        sentences = [s.strip().lower() for s in re.split(r"[\n\.;\•\-\*]", jd_text) if s.strip()]

        def classify_tier(term: str) -> str:
            term_lower = term.lower()
            for sentence in sentences:
                if term_lower in sentence:
                    # Check bonus first (subset of preferred language)
                    if any(cue in sentence for cue in self.BONUS_CUES):
                        return "bonus"
                    if any(cue in sentence for cue in self.PREFERRED_CUES):
                        return "preferred"
                    if any(cue in sentence for cue in self.REQUIRED_CUES):
                        return "required"
            return "required"  # Default: required

        required_skills, preferred_skills, bonus_skills = [], [], []
        required_tech, preferred_tech, bonus_tech = [], [], []

        for skill in all_skills:
            tier = classify_tier(skill)
            if tier == "bonus":
                bonus_skills.append(skill)
            elif tier == "preferred":
                preferred_skills.append(skill)
            else:
                required_skills.append(skill)

        for tech in all_tech:
            tier = classify_tier(tech)
            if tier == "bonus":
                bonus_tech.append(tech)
            elif tier == "preferred":
                preferred_tech.append(tech)
            else:
                required_tech.append(tech)

        return {
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "bonus_skills": bonus_skills,
            "required_technologies": required_tech,
            "preferred_technologies": preferred_tech,
            "bonus_technologies": bonus_tech,
            "languages": all_lang,
        }

    # -------------------------------------------------------------------
    # Semantic Score
    # -------------------------------------------------------------------

    def compute_semantic_score(self, candidate_text: str, jd_text: str) -> float:
        if not candidate_text.strip() or not jd_text.strip():
            return 0.0

        if self.model is not None:
            try:
                emb1 = self.model.encode([candidate_text])[0]
                emb2 = self.model.encode([jd_text])[0]
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                sim = float(np.dot(emb1, emb2) / (norm1 * norm2))
                return max(0.0, min(1.0, (sim + 1.0) / 2.0))
            except Exception:
                pass

        # Fallback: Jaccard similarity
        words1 = set(re.findall(r"\w+", candidate_text.lower()))
        words2 = set(re.findall(r"\w+", jd_text.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return float(len(intersection) / len(union))

    # -------------------------------------------------------------------
    # Snippet Finding for Evidence
    # -------------------------------------------------------------------

    def _find_snippet(self, term: str, text: str) -> str:
        if not text or not text.strip():
            return f"Specified as required criterion: '{term}'"

        sentences = [s.strip() for s in re.split(r"[\n\.;\•\-\*]", text) if len(s.strip()) > 5]
        term_lower = term.lower()

        # 1. Exact term match in sentence
        for sentence in sentences:
            if term_lower in sentence.lower():
                return sentence

        # 2. Check synonym terms for term
        syn_terms = [k for k, v in self.extractor.synonyms.items() if v.lower() == term_lower]
        for syn in syn_terms:
            for sentence in sentences:
                if syn.lower() in sentence.lower():
                    return sentence

        # 3. Substring match inside any word
        for sentence in sentences:
            words = sentence.lower().split()
            if any(term_lower in w for w in words):
                return sentence

        # 4. Fallback: Return first non-empty sentence or trimmed header text
        if sentences:
            return sentences[0]
        return text[:120].strip()

    # -------------------------------------------------------------------
    # Core Match Function — SINGLE SOURCE OF TRUTH
    # -------------------------------------------------------------------

    def compute_match(
        self,
        candidate_skills: List[str],
        candidate_tech: List[str],
        candidate_lang: List[str],
        candidate_text: str,
        jd_req_skills: List[str],
        jd_pref_skills: List[str],
        jd_req_tech: List[str],
        jd_pref_tech: List[str],
        jd_text: str,
        jd_bonus_skills: Optional[List[str]] = None,
        jd_bonus_tech: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        jd_bonus_skills = jd_bonus_skills or []
        jd_bonus_tech = jd_bonus_tech or []

        cand_skill_set = {s.lower() for s in candidate_skills}
        cand_tech_set = {t.lower() for t in candidate_tech}
        cand_lang_set = {l.lower() for l in candidate_lang}
        cand_all_set = cand_skill_set | cand_tech_set | cand_lang_set

        # ---- Skill Score with Related-Match Partial Credit ----
        # Direct match = full credit
        # Related match (ontology neighbor) = 0.5 partial credit
        # Missing = 0 credit
        def _score_requirements(jd_req: List[str], jd_pref: List[str], candidate_set: Set[str]) -> tuple:
            """Score requirements with Direct / Related / Missing tiers."""
            req_score_sum = 0.0
            req_direct, req_related, req_missing = [], [], []
            for term in jd_req:
                classification = self._classify_skill_match(term, candidate_set)
                if classification == "direct":
                    req_score_sum += 1.0
                    req_direct.append(term)
                elif classification == "related":
                    req_score_sum += 0.5  # Partial credit — not the same, but adjacent
                    req_related.append(term)
                else:
                    req_missing.append(term)
            req_ratio = req_score_sum / len(jd_req) if jd_req else 1.0

            pref_score_sum = 0.0
            pref_direct = []
            for term in jd_pref:
                classification = self._classify_skill_match(term, candidate_set)
                if classification == "direct":
                    pref_score_sum += 1.0
                    pref_direct.append(term)
                elif classification == "related":
                    pref_score_sum += 0.5
            pref_ratio = pref_score_sum / len(jd_pref) if jd_pref else 1.0

            if jd_req and jd_pref:
                combined = 0.8 * req_ratio + 0.2 * pref_ratio
            elif jd_req:
                combined = req_ratio
            elif jd_pref:
                combined = pref_ratio
            else:
                combined = 0.5

            return combined, req_direct, req_related, req_missing, pref_direct

        skill_score, req_skill_direct, req_skill_related, req_skill_missing, pref_skill_direct = \
            _score_requirements(jd_req_skills, jd_pref_skills, cand_skill_set | cand_lang_set)

        tech_score, req_tech_direct, req_tech_related, req_tech_missing, pref_tech_direct = \
            _score_requirements(jd_req_tech, jd_pref_tech, cand_tech_set)

        semantic_score = self.compute_semantic_score(candidate_text, jd_text)

        # Deterministic formula: 45% Skills · 30% Technologies · 25% Semantic
        skill_contribution = 0.45 * skill_score
        tech_contribution = 0.30 * tech_score
        semantic_contribution = 0.25 * semantic_score
        overall_score = skill_contribution + tech_contribution + semantic_contribution

        # ---- Match Classification ----
        all_jd_req = jd_req_skills + jd_req_tech
        all_jd_pref = jd_pref_skills + jd_pref_tech
        all_jd_bonus = jd_bonus_skills + jd_bonus_tech

        matched_required = list(set(req_skill_direct + req_tech_direct))
        matched_required_related = list(set(req_skill_related + req_tech_related))  # partial credit items
        matched_preferred = list(set(pref_skill_direct + pref_tech_direct))
        matched_bonus = [s for s in jd_bonus_skills + jd_bonus_tech if s.lower() in cand_all_set]

        missing_required_exact = list(set(req_skill_missing + req_tech_missing))  # no credit at all
        missing_preferred = [s for s in all_jd_pref if s.lower() not in cand_all_set
                             and self._classify_skill_match(s, cand_all_set) == "missing"]

        # Related competencies across ALL JD terms (for display)
        related_competencies = matched_required_related + [
            s for s in all_jd_pref
            if self._classify_skill_match(s, cand_all_set) == "related"
        ]

        # Hard gaps: required with zero credit (no direct, no related)
        hard_gaps = missing_required_exact
        has_hard_gaps = len(hard_gaps) > 0

        # Extra strengths: candidate skills not in JD at all
        all_jd_set = {s.lower() for s in all_jd_req + all_jd_pref + all_jd_bonus}
        extra_skills = [
            s for s in candidate_skills + candidate_tech
            if s.lower() not in all_jd_set
        ]

        # For backward compat — matched_skills = direct required + preferred
        matched_skills_compat = list(set(matched_required + matched_preferred))
        missing_skills_compat = missing_required_exact

        # Evidence snippets for matched terms
        evidence = []
        for term in (matched_required + matched_preferred)[:6]:
            evidence.append({
                "term": term,
                "candidate_snippet": self._find_snippet(term, candidate_text),
                "job_snippet": self._find_snippet(term, jd_text),
                "relationship": "Direct Match (Canonical)",
            })
        for term in matched_required_related[:3]:
            evidence.append({
                "term": term,
                "candidate_snippet": self._find_snippet(term, candidate_text),
                "job_snippet": self._find_snippet(term, jd_text),
                "relationship": "Related Competency (Partial Credit · 0.5x)",
            })

        # Score reasons
        reasons = []
        n_direct_req = len(matched_required)
        n_related_req = len(matched_required_related)
        n_total_req = len(all_jd_req)
        if not hard_gaps:
            reasons.append("All required criteria met (direct or related competency).")
        else:
            reasons.append(
                f"{n_direct_req} of {n_total_req} required criteria matched directly; "
                f"{n_related_req} via related competency; {len(hard_gaps)} hard gap(s)."
            )
        if matched_preferred:
            reasons.append(f"Meets {len(matched_preferred)} preferred criterion: {', '.join(matched_preferred[:3])}.")
        if matched_required_related:
            reasons.append(
                f"Related competency partial credit for: {', '.join(matched_required_related[:3])} "
                f"(0.5× credit — adjacent skill, not exact)."
            )
        if semantic_score >= 0.6:
            reasons.append("High textual similarity to job description.")
        elif semantic_score >= 0.3:
            reasons.append("Moderate contextual similarity to the job description.")
        else:
            reasons.append("Low general text similarity; fit driven by exact skill matches.")

        return {
            "skill_score": round(float(skill_score), 4),
            "tech_score": round(float(tech_score), 4),
            "semantic_score": round(float(semantic_score), 4),
            "overall_score": round(float(overall_score), 4),
            # Weighted contributions for UI transparency
            "weighted_contributions": {
                "skill": round(float(skill_contribution), 4),
                "tech": round(float(tech_contribution), 4),
                "semantic": round(float(semantic_contribution), 4),
                "weights": {"skill": 0.45, "tech": 0.30, "semantic": 0.25},
            },
            # Classified matches
            "matched_required": matched_required,
            "matched_required_related": matched_required_related,
            "matched_preferred": matched_preferred,
            "matched_bonus": matched_bonus,
            # For backward compat:
            "matched_skills": matched_skills_compat,
            "missing_required": missing_required_exact,
            "missing_preferred": missing_preferred,
            # For backward compat:
            "missing_skills": missing_required_exact,
            "related_competencies": related_competencies,
            "extra_skills": extra_skills,
            "hard_gaps": hard_gaps,
            "has_hard_gaps": has_hard_gaps,
            "evidence": evidence,
            "score_reasons": reasons,
        }

    # -------------------------------------------------------------------
    # Niche Role Recommendations
    # -------------------------------------------------------------------

    def recommend_roles(
        self,
        candidate_skills: List[str],
        candidate_tech: List[str],
        candidate_lang: List[str],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recommend niche roles from the role taxonomy using the same
        deterministic compute_match function.
        """
        results = []
        cand_skills_text = " ".join(candidate_skills + candidate_tech + candidate_lang)

        for role in self.job_roles:
            req_skills = role.get("required_skills", [])
            req_tech = role.get("required_technologies", [])
            all_req = req_skills + req_tech

            jd_text = (
                role.get("role_name", "")
                + " "
                + " ".join(req_skills)
                + " "
                + " ".join(role.get("preferred_skills", []))
                + " "
                + " ".join(req_tech)
            )

            match_res = self.compute_match(
                candidate_skills=candidate_skills,
                candidate_tech=candidate_tech,
                candidate_lang=candidate_lang,
                candidate_text=cand_skills_text,
                jd_req_skills=req_skills,
                jd_pref_skills=role.get("preferred_skills", []),
                jd_req_tech=req_tech,
                jd_pref_tech=role.get("preferred_technologies", []),
                jd_text=jd_text,
                jd_bonus_skills=[],
                jd_bonus_tech=[],
            )

            matched_req = match_res.get("matched_required", [])
            direct_ratio = len(matched_req) / len(all_req) if all_req else 0.0

            # Minimum evidence threshold check:
            # Must have at least 1 direct required match AND direct_ratio >= 0.35 (or at least 2 direct matches if many requirements)
            has_strong_signal = len(matched_req) >= 1 and (direct_ratio >= 0.35 or len(matched_req) >= 2)

            if has_strong_signal:
                results.append({
                    "id": role.get("id"),
                    "role_name": role.get("role_name"),
                    "domain": role.get("domain", ""),
                    "score": match_res["overall_score"],
                    "skill_score": match_res["skill_score"],
                    "tech_score": match_res["tech_score"],
                    "semantic_score": match_res["semantic_score"],
                    "matched_skills": match_res["matched_skills"],
                    "matched_required": match_res["matched_required"],
                    "matched_preferred": match_res["matched_preferred"],
                    "missing_skills": match_res["missing_required"],
                    "related_competencies": match_res["related_competencies"],
                    "extra_skills": match_res["extra_skills"],
                    "has_hard_gaps": match_res["has_hard_gaps"],
                    "score_reasons": match_res["score_reasons"],
                })

        results.sort(
            key=lambda x: (x["score"], x["skill_score"], x["tech_score"]),
            reverse=True,
        )
        return results[:top_n]
