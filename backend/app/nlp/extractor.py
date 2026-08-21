import json
import os
import re
from typing import Dict, List, Set, Any, Tuple
import spacy
from rapidfuzz import process, fuzz

class NLPExtractor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.nlp = None
        self._load_data()
        self._init_spacy()

    def _load_data(self):
        skills_path = os.path.join(self.data_dir, "skills.json")
        tech_path = os.path.join(self.data_dir, "technologies.json")
        lang_path = os.path.join(self.data_dir, "languages.json")
        synonyms_path = os.path.join(self.data_dir, "synonyms.json")

        with open(skills_path, "r", encoding="utf-8") as f:
            self.skills: List[str] = json.load(f)

        with open(tech_path, "r", encoding="utf-8") as f:
            self.technologies: List[str] = json.load(f)

        with open(lang_path, "r", encoding="utf-8") as f:
            self.languages: List[str] = json.load(f)

        with open(synonyms_path, "r", encoding="utf-8") as f:
            self.synonyms: Dict[str, str] = json.load(f)

        self.exact_skill_map = {s.lower(): s for s in self.skills}
        self.exact_tech_map = {t.lower(): t for t in self.technologies}
        self.exact_lang_map = {l.lower(): l for l in self.languages}

    def _init_spacy(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    # -------------------------------------------------------------------
    # Negation & Context Detection (deterministic NLP rules)
    # -------------------------------------------------------------------

    NEGATION_PATTERNS = [
        r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bnor\b",
        r"\bhave not\b", r"\bhaven't\b", r"\bdid not\b", r"\bdidn't\b",
        r"\bdo not\b", r"\bdon't\b", r"\bwithout\b", r"\bunfamiliar with\b",
        r"\bno experience\b", r"\bnot experienced\b", r"\bnot familiar\b",
    ]

    LEARNING_PATTERNS = [
        r"\bcurrently learning\b", r"\blearning\b", r"\bstudying\b",
        r"\bexploring\b", r"\bbeginning to\b", r"\bnew to\b",
        r"\bintroduction to\b", r"\bbasic knowledge\b",
    ]

    COURSEWORK_PATTERNS = [
        r"\bcourse\b", r"\bcoursework\b", r"\bclass\b", r"\blecture\b",
        r"\bacademic\b", r"\buniversity\b", r"\bcollege\b", r"\bsemester\b",
        r"\bassignment\b",
    ]

    PROJECT_PATTERNS = [
        r"\bproject\b", r"\bbuilt\b", r"\bdeveloped\b", r"\bcreated\b",
        r"\bimplemented\b", r"\bdesigned\b", r"\bprototype\b", r"\bhackathon\b",
        r"\bopen.?source\b",
    ]

    PROFESSIONAL_PATTERNS = [
        r"\bworked\b", r"\bwork\b", r"\bexperience\b", r"\bjob\b",
        r"\bposition\b", r"\brole\b", r"\bemployed\b", r"\bcompany\b",
        r"\bclient\b", r"\bproduction\b", r"\bprofessionally\b",
        r"\byears? of\b", r"\bfull.?time\b", r"\bpart.?time\b",
    ]

    INTERNSHIP_PATTERNS = [
        r"\bintern\b", r"\binternship\b", r"\bco-op\b", r"\bpractice\b",
        r"\bplacement\b",
    ]

    CURRENT_PATTERNS = [
        r"\bcurrently\b", r"\bpresent\b", r"\bongoing\b", r"\bstill\b",
        r"\bactive\b", r"\bnow\b",
    ]

    def _classify_context(self, sentence: str) -> str:
        """Classify the experience context of a sentence using deterministic NLP rules."""
        s = sentence.lower()

        # Negation check first — highest priority
        for pat in self.NEGATION_PATTERNS:
            if re.search(pat, s):
                return "not_experienced"

        # Learning
        for pat in self.LEARNING_PATTERNS:
            if re.search(pat, s):
                return "learning"

        # Coursework
        for pat in self.COURSEWORK_PATTERNS:
            if re.search(pat, s):
                return "coursework"

        # Internship
        for pat in self.INTERNSHIP_PATTERNS:
            if re.search(pat, s):
                return "internship"

        # Project
        for pat in self.PROJECT_PATTERNS:
            if re.search(pat, s):
                return "project"

        # Professional experience
        for pat in self.PROFESSIONAL_PATTERNS:
            if re.search(pat, s):
                return "professional"

        return "mention"

    def _classify_temporal(self, sentence: str) -> str:
        """Classify current/previous/learning temporal status."""
        s = sentence.lower()
        for pat in self.NEGATION_PATTERNS:
            if re.search(pat, s):
                return "not_experienced"
        for pat in self.CURRENT_PATTERNS:
            if re.search(pat, s):
                return "current"
        for pat in self.LEARNING_PATTERNS:
            if re.search(pat, s):
                return "learning"
        return "previous"

    def _find_source_sentence(self, term: str, text: str) -> str:
        """Find the sentence in text that contains the term."""
        sentences = [s.strip() for s in re.split(r"[.!?\n;]", text) if s.strip()]
        term_lower = term.lower()
        for sent in sentences:
            if term_lower in sent.lower():
                return sent
        return ""

    # -------------------------------------------------------------------
    # Main Extraction
    # -------------------------------------------------------------------

    def extract(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "skill": [],
                "technology": [],
                "language": [],
                "evidence": [],
                "pipeline_details": {
                    "total_entities": 0,
                    "noun_chunks": [],
                    "synonym_mappings": [],
                    "method": "Rule-based gazetteer lookup"
                }
            }

        doc = self.nlp(text)
        candidate_phrases: Set[str] = set()
        noun_chunks_list: List[str] = []

        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 1:
                candidate_phrases.add(token.text.strip())

        if hasattr(doc, "noun_chunks"):
            for chunk in doc.noun_chunks:
                c_text = chunk.text.strip()
                candidate_phrases.add(c_text)
                noun_chunks_list.append(c_text)

        words = [t.text for t in doc if not t.is_space]
        for n in range(1, 4):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                candidate_phrases.add(ngram.strip())

        extracted_skills: List[str] = []
        extracted_tech: List[str] = []
        extracted_lang: List[str] = []
        synonym_mappings: List[Dict[str, str]] = []
        # Evidence: per-entity provenance
        evidence_map: Dict[str, Dict[str, Any]] = {}

        seen_canonical: Set[str] = set()

        def add_entity(canonical_name: str, category: str, raw_matched: str):
            if canonical_name in seen_canonical:
                return
            seen_canonical.add(canonical_name)
            if category == "skill":
                extracted_skills.append(canonical_name)
            elif category == "technology":
                extracted_tech.append(canonical_name)
            elif category == "language":
                extracted_lang.append(canonical_name)

            if raw_matched.lower() != canonical_name.lower():
                synonym_mappings.append({
                    "raw_phrase": raw_matched,
                    "canonical": canonical_name,
                    "category": category
                })

            # Find source sentence and classify context
            source_sent = self._find_source_sentence(raw_matched, text)
            if not source_sent:
                source_sent = self._find_source_sentence(canonical_name, text)

            context = self._classify_context(source_sent) if source_sent else "mention"
            temporal = self._classify_temporal(source_sent) if source_sent else "previous"

            evidence_map[canonical_name] = {
                "canonical": canonical_name,
                "raw_phrase": raw_matched,
                "category": category,
                "source_sentence": source_sent or f"Inferred from '{raw_matched}'",
                "experience_context": context,
                "temporal_status": temporal,
            }

        sorted_phrases = sorted(list(candidate_phrases), key=lambda p: len(p), reverse=True)

        for phrase in sorted_phrases:
            phrase_clean = phrase.strip().lower()
            if not phrase_clean or len(phrase_clean) < 2:
                continue

            # 1. Exact Synonym Match
            if phrase_clean in self.synonyms:
                canonical = self.synonyms[phrase_clean]
                if canonical in self.skills or canonical.lower() in self.exact_skill_map:
                    add_entity(self.exact_skill_map.get(canonical.lower(), canonical), "skill", phrase)
                elif canonical in self.technologies or canonical.lower() in self.exact_tech_map:
                    add_entity(self.exact_tech_map.get(canonical.lower(), canonical), "technology", phrase)
                elif canonical in self.languages or canonical.lower() in self.exact_lang_map:
                    add_entity(self.exact_lang_map.get(canonical.lower(), canonical), "language", phrase)
                continue

            # 2. Exact Gazetteer Match
            if phrase_clean in self.exact_skill_map:
                add_entity(self.exact_skill_map[phrase_clean], "skill", phrase)
                continue
            if phrase_clean in self.exact_tech_map:
                add_entity(self.exact_tech_map[phrase_clean], "technology", phrase)
                continue
            if phrase_clean in self.exact_lang_map:
                add_entity(self.exact_lang_map[phrase_clean], "language", phrase)
                continue

            # 3. High-Confidence Fuzzy Match (Threshold >= 92)
            if len(phrase_clean) >= 4:
                match_skill = process.extractOne(phrase_clean, list(self.exact_skill_map.keys()), scorer=fuzz.ratio)
                if match_skill and match_skill[1] >= 92:
                    add_entity(self.exact_skill_map[match_skill[0]], "skill", phrase)
                    continue

                match_tech = process.extractOne(phrase_clean, list(self.exact_tech_map.keys()), scorer=fuzz.ratio)
                if match_tech and match_tech[1] >= 92:
                    add_entity(self.exact_tech_map[match_tech[0]], "technology", phrase)
                    continue

                match_lang = process.extractOne(phrase_clean, list(self.exact_lang_map.keys()), scorer=fuzz.ratio)
                if match_lang and match_lang[1] >= 92:
                    add_entity(self.exact_lang_map[match_lang[0]], "language", phrase)
                    continue

        # Filter out negated entities from results
        negated = {k for k, v in evidence_map.items() if v["experience_context"] == "not_experienced"}

        extracted_skills = [s for s in extracted_skills if s not in negated]
        extracted_tech = [t for t in extracted_tech if t not in negated]
        extracted_lang = [l for l in extracted_lang if l not in negated]

        # Build ordered evidence list (only non-negated)
        evidence_list = [
            v for k, v in evidence_map.items() if k not in negated
        ]
        negated_list = [
            v for k, v in evidence_map.items() if k in negated
        ]

        total_count = len(extracted_skills) + len(extracted_tech) + len(extracted_lang)

        return {
            "skill": extracted_skills,
            "technology": extracted_tech,
            "language": extracted_lang,
            "evidence": evidence_list,
            "negated_entities": negated_list,
            "pipeline_details": {
                "total_entities": total_count,
                "noun_chunks": list(set(noun_chunks_list))[:10],
                "synonym_mappings": synonym_mappings,
                "method": "spaCy Noun Chunking + Gazetteer Lookup + Synonym Normalization + Context Classification"
            }
        }
