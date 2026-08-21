from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field
import json

class Candidate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    contact_info: Optional[str] = None
    raw_text: str
    resume_file_path: Optional[str] = None
    skills_json: str = "[]"
    technologies_json: str = "[]"
    languages_json: str = "[]"
    experience: Optional[str] = None
    education: Optional[str] = None
    projects: Optional[str] = None
    text_hash: Optional[str] = Field(default=None, index=True)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def skills(self) -> List[str]:
        return json.loads(self.skills_json)

    @skills.setter
    def skills(self, value: List[str]):
        self.skills_json = json.dumps(value)

    @property
    def technologies(self) -> List[str]:
        return json.loads(self.technologies_json)

    @technologies.setter
    def technologies(self, value: List[str]):
        self.technologies_json = json.dumps(value)

    @property
    def languages(self) -> List[str]:
        return json.loads(self.languages_json)

    @languages.setter
    def languages(self, value: List[str]):
        self.languages_json = json.dumps(value)


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: str
    required_skills_json: str = "[]"
    preferred_skills_json: str = "[]"
    required_technologies_json: str = "[]"
    preferred_technologies_json: str = "[]"
    languages_json: str = "[]"
    experience_requirements: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def required_skills(self) -> List[str]:
        return json.loads(self.required_skills_json)

    @required_skills.setter
    def required_skills(self, value: List[str]):
        self.required_skills_json = json.dumps(value)

    @property
    def preferred_skills(self) -> List[str]:
        return json.loads(self.preferred_skills_json)

    @preferred_skills.setter
    def preferred_skills(self, value: List[str]):
        self.preferred_skills_json = json.dumps(value)

    @property
    def required_technologies(self) -> List[str]:
        return json.loads(self.required_technologies_json)

    @required_technologies.setter
    def required_technologies(self, value: List[str]):
        self.required_technologies_json = json.dumps(value)

    @property
    def preferred_technologies(self) -> List[str]:
        return json.loads(self.preferred_technologies_json)

    @preferred_technologies.setter
    def preferred_technologies(self, value: List[str]):
        self.preferred_technologies_json = json.dumps(value)


class Match(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: int = Field(index=True)
    job_id: int = Field(index=True)
    skill_score: float
    tech_score: float
    semantic_score: float
    overall_score: float
    matched_skills_json: str = "[]"
    missing_skills_json: str = "[]"
    explanation: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
