"use client";

import { useEffect, useState } from "react";
import {
  fetchCandidates,
  fetchJobs,
  askRecruiterAssistant,
  CandidateProfile,
  JobProfile,
  AssistantChatResponse,
  EvidenceCitation
} from "@/lib/api";
import {
  Bot,
  Users,
  Briefcase,
  Send,
  Sparkles,
  FileText,
  CheckCircle2,
  AlertTriangle,
  BookOpen,
  Terminal,
  RefreshCw,
  HelpCircle,
  ShieldCheck,
  ChevronRight,
  Database
} from "lucide-react";

interface ChatMessage {
  id: string;
  sender: "user" | "assistant";
  text: string;
  responseObj?: AssistantChatResponse;
  timestamp: string;
}

export default function AssistantPage() {
  const [candidates, setCandidates] = useState<CandidateProfile[]>([]);
  const [jobs, setJobs] = useState<JobProfile[]>([]);

  const [selectedCandidateId, setSelectedCandidateId] = useState<number | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);

  const [inputQuestion, setInputQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [candList, jobList] = await Promise.all([fetchCandidates(), fetchJobs()]);
      setCandidates(candList);
      setJobs(jobList);

      if (candList.length > 0) {
        setSelectedCandidateId(candList[0].id);
      }
      if (jobList.length > 0) {
        setSelectedJobId(jobList[0].id);
      }
    } catch (err) {
      console.error("Failed to load initial assistant data", err);
    }
  };

  const handleSendQuestion = async (customQ?: string) => {
    const query = customQ || inputQuestion;
    if (!query.trim() || !selectedCandidateId) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      sender: "user",
      text: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!customQ) setInputQuestion("");
    setLoading(true);

    try {
      const resp = await askRecruiterAssistant({
        candidate_id: selectedCandidateId,
        job_id: selectedJobId || undefined,
        question: query
      });

      const aiMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: "assistant",
        text: resp.answer,
        responseObj: resp,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: "assistant",
        text: `Error connecting to Assistant: ${err.message || "Failed to process query"}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const selectedCandidate = candidates.find((c) => c.id === selectedCandidateId);
  const selectedJob = jobs.find((j) => j.id === selectedJobId);

  const presetQuestions = [
    { label: "Why is candidate a good fit?", query: "Why is this candidate a good fit for this position?" },
    { label: "Explain score breakdown & gaps", query: "Why did the candidate receive this match score? What are the key gaps?" },
    { label: "Show RAG & AI Agent evidence", query: "Show me direct evidence of candidate's RAG, Vector Search, and AI Agent experience." },
    { label: "Missing required skills", query: "What mandatory required skills or technologies are missing from this profile?" },
    { label: "Job mandatory requirements", query: "What are the core mandatory requirements for this job?" }
  ];

  return (
    <div className="space-y-8">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-hairline pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono bg-emerald-100 text-emerald-800 px-2.5 py-0.5 rounded border border-emerald-300 font-semibold uppercase">
              Part 3 · Local RAG + Llama 3:8B
            </span>
          </div>
          <h1 className="text-3xl font-semibold text-ink tracking-tight">Recruiter Intelligence Assistant</h1>
          <p className="text-sm text-steel mt-1">
            Grounded Q&A powered by ChromaDB vector retrieval, Part 2 deterministic match scores, and local Llama 3:8B.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs text-steel font-mono bg-canvas p-2.5 rounded-xl border border-hairline">
          <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>Scores Fixed by Part 2 · Zero LLM Score Alteration</span>
        </div>
      </div>

      {/* Top Context Selector Card */}
      <div className="card-grafbase p-6 bg-canvas/60 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Candidate Dropdown */}
          <div>
            <label className="block text-xs font-medium text-steel mb-1.5 flex items-center gap-1.5">
              <Users className="w-4 h-4 text-teal-600" />
              <span>Select Candidate Context *</span>
            </label>
            <select
              value={selectedCandidateId || ""}
              onChange={(e) => setSelectedCandidateId(Number(e.target.value))}
              className="w-full p-2.5 rounded-lg border border-hairline bg-canvas text-xs font-semibold text-ink focus:outline-none focus:ring-2 focus:ring-ink"
            >
              {candidates.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} (Candidate #{c.id} · {c.skills.length + c.technologies.length} Extracted Entities)
                </option>
              ))}
            </select>
          </div>

          {/* Job Context Dropdown */}
          <div>
            <label className="block text-xs font-medium text-steel mb-1.5 flex items-center gap-1.5">
              <Briefcase className="w-4 h-4 text-sky-600" />
              <span>Select Job Context *</span>
            </label>
            <select
              value={selectedJobId || ""}
              onChange={(e) => setSelectedJobId(Number(e.target.value))}
              className="w-full p-2.5 rounded-lg border border-hairline bg-canvas text-xs font-semibold text-ink focus:outline-none focus:ring-2 focus:ring-ink"
            >
              <option value="">-- General Candidate Analysis (No Job Selected) --</option>
              {jobs.map((j) => (
                <option key={j.id} value={j.id}>
                  {j.title} (Job #{j.id})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Selected Context Summary Pill */}
        {selectedCandidate && (
          <div className="p-3.5 rounded-xl bg-canvas border border-hairline flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 text-xs">
            <div className="space-y-1">
              <span className="font-semibold text-ink">Active Analysis Pair:</span>
              <p className="text-steel font-mono">
                Candidate: <strong className="text-ink">{selectedCandidate.name}</strong>
                {selectedJob ? (
                  <> vs Job: <strong className="text-ink">{selectedJob.title}</strong></>
                ) : (
                  <> (Standalone Candidate Resume Analysis)</>
                )}
              </p>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <span className="badge-sky px-2.5 py-1 rounded text-[11px] font-mono">
                {selectedCandidate.skills.length} Skills · {selectedCandidate.technologies.length} Tech
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Preset Quick Prompt Chips */}
      <div className="space-y-2">
        <span className="text-xs font-medium text-steel flex items-center gap-1">
          <HelpCircle className="w-3.5 h-3.5" />
          <span>Quick Recruiter Questions:</span>
        </span>

        <div className="flex flex-wrap gap-2">
          {presetQuestions.map((pq, idx) => (
            <button
              key={idx}
              onClick={() => handleSendQuestion(pq.query)}
              disabled={loading}
              className="text-xs bg-canvas/80 hover:bg-canvas text-ink border border-hairline hover:border-steel px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 text-left font-medium"
            >
              <ChevronRight className="w-3 h-3 text-teal-600" />
              <span>{pq.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Chat Stream Container */}
      <div className="card-grafbase p-6 min-h-[400px] flex flex-col justify-between space-y-6">
        {/* Messages */}
        <div className="space-y-6">
          {messages.length === 0 ? (
            <div className="py-16 text-center space-y-4 text-steel">
              <Bot className="w-12 h-12 text-ash mx-auto stroke-[1.5]" />
              <h3 className="text-base font-semibold text-ink">Ask Recruiter Intelligence Assistant</h3>
              <p className="text-xs max-w-md mx-auto">
                Ask questions about candidate fit, hard gaps, resume project evidence, or job requirements.
                Answers are grounded in ChromaDB vector search and local Llama 3:8B.
              </p>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className="space-y-3">
                {/* User Message */}
                {msg.sender === "user" ? (
                  <div className="flex justify-end">
                    <div className="bg-ink text-white p-4 rounded-2xl max-w-2xl text-xs space-y-1 shadow-sm">
                      <div className="flex justify-between items-center text-[10px] text-zinc-300 font-mono border-b border-zinc-700 pb-1 mb-1">
                        <span>Recruiter Query</span>
                        <span>{msg.timestamp}</span>
                      </div>
                      <p className="font-medium text-sm">{msg.text}</p>
                    </div>
                  </div>
                ) : (
                  /* Assistant Response Card */
                  <div className="flex justify-start">
                    <div className="bg-canvas border border-hairline p-6 rounded-2xl max-w-3xl w-full text-xs space-y-5 shadow-sm">
                      {/* Model & Source Header */}
                      <div className="flex items-center justify-between border-b border-hairline pb-3">
                        <div className="flex items-center gap-2">
                          <div className="w-6 h-6 rounded-md bg-emerald-700 text-white flex items-center justify-center font-bold text-xs font-mono">
                            AI
                          </div>
                          <div>
                            <span className="font-semibold text-ink text-xs block">Recruiter Intelligence Assistant</span>
                            <span className="text-[10px] font-mono text-steel">
                              Model: {msg.responseObj?.model_used || "Llama 3:8B"}
                            </span>
                          </div>
                        </div>

                        <span className="text-[10px] font-mono text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                          Grounded Response
                        </span>
                      </div>

                      {/* Answer Body */}
                      <div className="prose prose-sm max-w-none text-ink text-xs leading-relaxed space-y-2 whitespace-pre-line font-mono">
                        {msg.text ? msg.text.replace(/\\#/g, "#").replace(/\\\*/g, "*").replace(/\\_/g, "_").replace(/\\`/g, "`").replace(/\bsvg\b/gi, "").trim() : ""}
                      </div>

                      {/* Expandable Evidence & Sources Accordion */}
                      {(msg.responseObj?.evidence_citations && msg.responseObj.evidence_citations.length > 0) || msg.responseObj?.deterministic_match ? (
                        <div className="pt-3 border-t border-hairline space-y-3">
                          <details className="group">
                            <summary className="cursor-pointer text-xs font-semibold text-steel hover:text-ink flex items-center justify-between p-2 rounded-lg bg-canvas/40 border border-hairline transition-all">
                              <span className="flex items-center gap-1.5 font-mono">
                                <BookOpen className="w-4 h-4 text-sky-600" />
                                <span>Evidence & Sources ({msg.responseObj.evidence_citations?.length || 0} Chunks)</span>
                              </span>
                              <span className="text-[10px] text-steel font-mono group-open:rotate-180 transition-transform">▼</span>
                            </summary>

                            <div className="mt-3 space-y-3 pt-2">
                              {/* Citations List */}
                              {msg.responseObj?.evidence_citations && msg.responseObj.evidence_citations.length > 0 && (
                                <div className="grid grid-cols-1 gap-2">
                                  {msg.responseObj.evidence_citations.map((cit) => (
                                    <div key={cit.citation_num} className="p-3 rounded-lg bg-canvas/60 border border-hairline space-y-1">
                                      <div className="flex items-center justify-between text-[11px] font-mono">
                                        <span className="font-semibold text-ink flex items-center gap-1">
                                          <span className="w-4 h-4 rounded-full bg-sky-100 text-sky-800 flex items-center justify-center text-[10px] font-bold">
                                            {cit.citation_num}
                                          </span>
                                          <span>{cit.source}</span>
                                        </span>
                                        <span className="text-steel bg-canvas px-2 py-0.5 rounded border border-hairline">
                                          Section: {cit.section}
                                        </span>
                                      </div>
                                      <p className="text-[11px] text-steel italic font-mono pl-5">
                                        "{cit.snippet}"
                                      </p>
                                    </div>
                                  ))}
                                </div>
                              )}

                              {/* Part 2 Ground Truth Match Summary Badge */}
                              {msg.responseObj?.deterministic_match && (
                                <div className="p-3 rounded-xl bg-zinc-900 text-zinc-100 space-y-2 text-[11px] font-mono">
                                  <div className="flex justify-between items-center">
                                    <span className="text-emerald-400 font-semibold flex items-center gap-1">
                                      <ShieldCheck className="w-3.5 h-3.5" />
                                      <span>Part 2 Deterministic Match Summary (Ground Truth)</span>
                                    </span>
                                    <span className="text-zinc-400">
                                      Score: {Math.round(msg.responseObj.deterministic_match.overall_score * 100)}%
                                    </span>
                                  </div>

                                  <div className="flex flex-wrap gap-4 text-zinc-300 text-[10px]">
                                    <span>Skill Score: {Math.round(msg.responseObj.deterministic_match.skill_score * 100)}%</span>
                                    <span>Tech Score: {Math.round(msg.responseObj.deterministic_match.tech_score * 100)}%</span>
                                    <span>Semantic Score: {Math.round(msg.responseObj.deterministic_match.semantic_score * 100)}%</span>
                                  </div>
                                </div>
                              )}
                            </div>
                          </details>
                        </div>
                      ) : null}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}

          {loading && (
            <div className="flex items-center gap-3 text-xs text-steel font-mono p-4 bg-canvas/40 rounded-xl border border-hairline animate-pulse">
              <div className="w-4 h-4 border-2 border-emerald-600 border-t-transparent rounded-full animate-spin" />
              <span>Searching ChromaDB vector store & querying Llama 3:8B...</span>
            </div>
          )}
        </div>

        {/* Input Box */}
        <div className="pt-4 border-t border-hairline space-y-2">
          <div className="flex gap-2">
            <textarea
              rows={2}
              placeholder="Ask the Recruiter Assistant about candidate fit, hard gaps, or project evidence..."
              value={inputQuestion}
              onChange={(e) => setInputQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendQuestion();
                }
              }}
              className="flex-1 p-3 rounded-xl border border-hairline bg-canvas text-xs focus:outline-none focus:ring-2 focus:ring-ink"
            />
            <button
              onClick={() => handleSendQuestion()}
              disabled={loading || !inputQuestion.trim() || !selectedCandidateId}
              className="btn-primary px-5 flex items-center justify-center gap-2 text-xs shrink-0 self-end h-10"
            >
              <span>{loading ? "Thinking..." : "Ask"}</span>
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="flex justify-between items-center text-[10px] text-steel font-mono">
            <span>Press Enter to send · Grounded in ChromaDB & Llama 3:8B</span>
            {messages.length > 0 && (
              <button
                onClick={() => setMessages([])}
                className="hover:underline text-rose-600"
              >
                Clear Chat
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
