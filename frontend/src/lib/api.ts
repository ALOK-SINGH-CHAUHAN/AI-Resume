const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PipelineDetails {
  total_entities: number;
  noun_chunks: string[];
  synonym_mappings: Array<{ raw_phrase: string; canonical: string; category: string }>;
  method: string;
}

export interface EvidenceEntry {
  canonical: string;
  raw_phrase: string;
  category: string;
  source_sentence: string;
  experience_context: "professional" | "internship" | "project" | "coursework" | "learning" | "mention" | "not_experienced";
  temporal_status: "current" | "previous" | "learning" | "not_experienced";
}

export interface ExtractedEntities {
  skill: string[];
  technology: string[];
  language: string[];
  evidence?: EvidenceEntry[];
  negated_entities?: EvidenceEntry[];
  pipeline_details?: PipelineDetails;
}

export interface CandidateProfile {
  id: number;
  name: string;
  contact_info?: string;
  raw_text: string;
  skills: string[];
  technologies: string[];
  languages: string[];
  created_at: string;
}

export interface JobProfile {
  id: number;
  title: string;
  description: string;
  required_skills: string[];
  preferred_skills: string[];
  required_technologies: string[];
  preferred_technologies: string[];
  created_at: string;
}


export interface MatchEvidence {
  term: string;
  candidate_snippet: string;
  job_snippet: string;
  relationship: string;
}

export interface MatchResult {
  candidate_id: number;
  candidate_name?: string;
  job_id: number;
  job_title?: string;
  skill_score: number;
  tech_score: number;
  semantic_score: number;
  overall_score: number;
  matched_skills: string[];
  matched_required?: string[];
  matched_preferred?: string[];
  matched_bonus?: string[];
  missing_skills: string[];
  missing_required?: string[];
  missing_preferred?: string[];
  related_competencies?: string[];
  extra_skills?: string[];
  hard_gaps?: string[];
  has_hard_gaps?: boolean;
  evidence?: MatchEvidence[];
  score_reasons?: string[];
}

export interface RoleRecommendation {
  id: string;
  role_name: string;
  domain?: string;
  score: number;
  skill_score: number;
  tech_score: number;
  semantic_score?: number;
  matched_skills: string[];
  matched_required?: string[];
  matched_preferred?: string[];
  missing_skills: string[];
  related_competencies?: string[];
  has_hard_gaps?: boolean;
  score_reasons?: string[];
}

export interface RemotiveJobMatch {
  remotive_id: number;
  title: string;
  company_name: string;
  url: string;
  category: string;
  job_type: string;
  location: string;
  description_snippet: string;
  overall_score: number;
  skill_score: number;
  tech_score: number;
  semantic_score: number;
  matched_skills: string[];
  matched_required: string[];
  matched_preferred: string[];
  missing_skills: string[];
  hard_gaps: string[];
  has_hard_gaps: boolean;
  score_reasons: string[];
}

export interface RemotiveRecommendationResponse {
  candidate_id: number;
  candidate_name: string;
  total_found: number;
  recommended_jobs: RemotiveJobMatch[];
}

export async function checkApiHealth(): Promise<"online" | "connecting" | "offline"> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: 'no-store' });
    if (res.ok) {
      const data = await res.json().catch(() => ({}));
      if (data.status === "ok") return "online";
    }
    return "offline";
  } catch {
    return "offline";
  }
}

export async function extractEntities(text: string): Promise<ExtractedEntities> {
  const res = await fetch(`${API_URL}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Extraction failed" }));
    throw new Error(err.detail || "Extraction failed");
  }
  return res.json();
}

export async function saveCandidate(candidate: {
  name: string;
  contact_info?: string;
  raw_text: string;
  skills: string[];
  technologies: string[];
  languages: string[];
}): Promise<CandidateProfile & { is_duplicate?: boolean; message?: string }> {
  const res = await fetch(`${API_URL}/candidate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(candidate),
  });
  if (!res.ok) throw new Error("Failed to save candidate profile");
  return res.json();
}

export async function fetchCandidates(): Promise<CandidateProfile[]> {
  const res = await fetch(`${API_URL}/candidates`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Failed to fetch candidates");
  return res.json();
}

export async function fetchCandidateById(id: number): Promise<CandidateProfile> {
  const res = await fetch(`${API_URL}/candidate/${id}`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Candidate not found");
  return res.json();
}

export async function deleteCandidate(id: number): Promise<void> {
  const res = await fetch(`${API_URL}/candidate/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete candidate");
}

export async function createJob(job: { title: string; description: string }): Promise<JobProfile> {
  const res = await fetch(`${API_URL}/job`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(job),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Job creation failed" }));
    throw new Error(err.detail || "Job creation failed");
  }
  return res.json();
}

export async function fetchJobs(): Promise<JobProfile[]> {
  const res = await fetch(`${API_URL}/jobs`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Failed to fetch jobs");
  return res.json();
}

export async function fetchJobById(id: number): Promise<JobProfile> {
  const res = await fetch(`${API_URL}/job/${id}`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Job not found");
  return res.json();
}

export async function matchCandidateAndJob(candidateId: number, jobId: number): Promise<MatchResult> {
  const res = await fetch(`${API_URL}/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate_id: candidateId, job_id: jobId }),
  });
  if (!res.ok) throw new Error("Match computation failed");
  return res.json();
}

export async function recommendRoles(candidateId: number, topN = 5): Promise<RoleRecommendation[]> {
  const res = await fetch(`${API_URL}/recommend-roles`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate_id: candidateId, top_n: topN }),
  });
  if (!res.ok) throw new Error("Role recommendation failed");
  const data = await res.json();
  return data.roles;
}

export async function fetchCandidateRankingsForJob(jobId: number): Promise<{
  job_id: number;
  job_title: string;
  rankings: Array<{
    candidate_id: number;
    candidate_name: string;
    overall_score: number;
    skill_score: number;
    tech_score: number;
    semantic_score: number;
    matched_skills: string[];
    missing_skills: string[];
    extra_skills?: string[];
  }>;
}> {
  const res = await fetch(`${API_URL}/matches/${jobId}`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Candidate ranking failed");
  return res.json();
}

export async function uploadResumeFile(file: File): Promise<{
  filename: string;
  extracted_text: string;
  preview: string;
  char_count: number;
}> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_URL}/resume/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "File upload failed" }));
    throw new Error(err.detail || "File upload failed");
  }
  return res.json();
}

export async function fetchRemotiveRecommendedJobs(candidateId: number): Promise<RemotiveRecommendationResponse> {
  const res = await fetch(`${API_URL}/remotive/recommended/${candidateId}`, { cache: 'no-store' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Failed to fetch Remotive jobs." }));
    throw new Error(err.detail || "Failed to fetch Remotive jobs.");
  }
  return res.json();
}

export interface EvidenceCitation {
  citation_num: number;
  source: string;
  section: string;
  snippet: string;
}

export interface AssistantChatResponse {
  answer: string;
  evidence_citations: EvidenceCitation[];
  deterministic_match: MatchResult | null;
  retrieved_chunks: Array<{
    text: string;
    section: string;
    source: string;
    similarity_score: number;
  }>;
  model_used: string;
}

export async function askRecruiterAssistant(params: {
  candidate_id: number;
  job_id?: number | null;
  question: string;
}): Promise<AssistantChatResponse> {
  const res = await fetch(`${API_URL}/assistant/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Assistant chat failed" }));
    throw new Error(err.detail || "Assistant chat failed");
  }
  return res.json();
}
