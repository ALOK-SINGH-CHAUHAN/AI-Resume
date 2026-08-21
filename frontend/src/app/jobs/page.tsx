"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  fetchJobs,
  createJob,
  fetchCandidates,
  fetchRemotiveRecommendedJobs,
  JobProfile,
  CandidateProfile,
  RemotiveJobMatch
} from "@/lib/api";
import {
  Briefcase,
  Plus,
  ArrowRight,
  Globe,
  ExternalLink,
  CheckCircle2,
  XCircle,
  Users,
  Search,
  RefreshCw,
  Building,
  MapPin,
  Sparkles
} from "lucide-react";

export default function JobsPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<"library" | "recommended">("library");
  const [searchQuery, setSearchQuery] = useState("");

  // Local Job Library State
  const [jobs, setJobs] = useState<JobProfile[]>([]);
  const [loadingJobs, setLoadingJobs] = useState(true);

  // Form Modal State
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Recommended Jobs (Remotive API) State
  const [candidates, setCandidates] = useState<CandidateProfile[]>([]);
  const [selectedCandidateId, setSelectedCandidateId] = useState<number | null>(null);
  const [recommendedJobs, setRecommendedJobs] = useState<RemotiveJobMatch[]>([]);
  const [loadingRemotive, setLoadingRemotive] = useState(false);
  const [importingId, setImportingId] = useState<number | null>(null);

  const loadJobs = async () => {
    setLoadingJobs(true);
    try {
      const data = await fetchJobs();
      setJobs(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingJobs(false);
    }
  };

  const loadCandidates = async () => {
    try {
      const candList = await fetchCandidates();
      setCandidates(candList);
      if (candList.length > 0) {
        // Default to candidate with richest skills or first
        const bestCand = candList.find((c) => c.skills.length > 3) || candList[0];
        setSelectedCandidateId(bestCand.id);
        // Automatically pre-fetch Remotive jobs for default candidate
        handleFetchRemotive(bestCand.id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadJobs();
    loadCandidates();
  }, []);

  // Fetch Remotive jobs when selected candidate changes
  useEffect(() => {
    if (selectedCandidateId) {
      handleFetchRemotive(selectedCandidateId);
    }
  }, [selectedCandidateId]);

  const handleFetchRemotive = async (candId: number) => {
    setLoadingRemotive(true);
    try {
      const data = await fetchRemotiveRecommendedJobs(candId);
      setRecommendedJobs(data.recommended_jobs);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingRemotive(false);
    }
  };

  const handleCreateJob = async () => {
    if (!title.trim() || !description.trim()) {
      setError("Please fill out both job title and description.");
      return;
    }
    setError(null);
    setCreating(true);
    try {
      await createJob({ title, description });
      setShowCreateModal(false);
      setTitle("");
      setDescription("");
      loadJobs();
    } catch (err: any) {
      setError(err.message || "Failed to create job.");
    } finally {
      setCreating(false);
    }
  };

  const handleImportRemotiveJob = async (remotiveJob: RemotiveJobMatch) => {
    setImportingId(remotiveJob.remotive_id);
    try {
      await createJob({
        title: `${remotiveJob.title} (${remotiveJob.company_name})`,
        description: remotiveJob.description_snippet
      });
      await loadJobs();
      setActiveTab("library");
    } catch (err) {
      console.error(err);
    } finally {
      setImportingId(null);
    }
  };

  const fillPreset = () => {
    setTitle("Senior Machine Learning Engineer");
    setDescription(
      "We are looking for a Senior Machine Learning Engineer with required skills in Machine Learning, Deep Learning, and Python. Required technologies: TensorFlow, PyTorch, and CNN. Preferred skills: Computer Vision and Natural Language Processing. Bonus: Docker and AWS experience."
    );
  };

  // Filter local jobs by search query
  const filteredLocalJobs = jobs.filter((j) => {
    const q = searchQuery.toLowerCase();
    return (
      !q ||
      j.title.toLowerCase().includes(q) ||
      j.description.toLowerCase().includes(q) ||
      j.required_skills.some((s) => s.toLowerCase().includes(q)) ||
      j.required_technologies.some((t) => t.toLowerCase().includes(q))
    );
  });

  // Filter recommended jobs by search query
  const filteredRemotiveJobs = recommendedJobs.filter((rj) => {
    const q = searchQuery.toLowerCase();
    return (
      !q ||
      rj.title.toLowerCase().includes(q) ||
      rj.company_name.toLowerCase().includes(q) ||
      rj.matched_skills.some((ms) => ms.toLowerCase().includes(q))
    );
  });

  const selectedCandidate = candidates.find((c) => c.id === selectedCandidateId);

  return (
    <div className="space-y-8">
      {/* Top Header & New Job Action */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-hairline pb-6">
        <div>
          <h1 className="text-3xl font-semibold text-ink tracking-tight">Job Library</h1>
          <p className="text-sm text-steel mt-1">
            Manage local job descriptions or discover live remote opportunities matched against candidates via Remotive API.
          </p>
        </div>

        <button
          onClick={() => setShowCreateModal(true)}
          className="btn-primary flex items-center gap-2 text-xs"
        >
          <Plus className="w-4 h-4" />
          <span>New Job Description</span>
        </button>
      </div>

      {/* Prominent Primary Sub-Tabs Navigation */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-hairline pb-1">
        <div className="flex gap-8 text-base font-semibold">
          <button
            onClick={() => setActiveTab("library")}
            className={`pb-3 flex items-center gap-2 transition-all border-b-2 ${
              activeTab === "library"
                ? "border-ink text-ink font-bold"
                : "border-transparent text-steel hover:text-ink font-medium"
            }`}
          >
            <Briefcase className="w-5 h-5 text-teal-600" />
            <span>Job Library ({jobs.length} Local JDs)</span>
          </button>

          <button
            onClick={() => setActiveTab("recommended")}
            className={`pb-3 flex items-center gap-2 transition-all border-b-2 ${
              activeTab === "recommended"
                ? "border-emerald-600 text-emerald-800 font-bold"
                : "border-transparent text-steel hover:text-ink font-medium"
            }`}
          >
            <Globe className="w-5 h-5 text-emerald-600" />
            <span>Recommended Jobs ({recommendedJobs.length} Remotive Live Listings)</span>
            <span className="text-[10px] uppercase font-mono bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded border border-emerald-300">
              Live API
            </span>
          </button>
        </div>

        {/* Search Input Box */}
        <div className="relative w-full sm:w-72 mb-2">
          <Search className="w-4 h-4 text-steel absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search jobs by title or skill..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 rounded-lg border border-hairline bg-canvas text-xs focus:outline-none focus:ring-2 focus:ring-ink font-mono"
          />
        </div>
      </div>

      {/* TAB 1: JOB LIBRARY (Local Recruiter Workflow) */}
      {activeTab === "library" && (
        <div className="space-y-6">
          {/* Helper Banner pointing to Recommended Jobs */}
          <div className="p-4 rounded-xl bg-canvas/60 border border-hairline flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
            <div className="space-y-0.5">
              <span className="text-xs font-semibold text-ink flex items-center gap-1.5">
                <Globe className="w-4 h-4 text-emerald-600" />
                <span>Live Remote Opportunity Discovery Available</span>
              </span>
              <p className="text-xs text-steel">
                Looking for real live remote job listings from Remotive matched against candidates? Switch to the Recommended Jobs tab.
              </p>
            </div>
            <button
              onClick={() => setActiveTab("recommended")}
              className="text-xs font-semibold text-emerald-700 hover:underline flex items-center gap-1 shrink-0"
            >
              <span>Explore Live Remotive Jobs</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {loadingJobs ? (
            <div className="py-16 text-center text-steel font-mono">Loading job descriptions...</div>
          ) : filteredLocalJobs.length === 0 ? (
            <div className="card-grafbase p-12 text-center space-y-4">
              <Briefcase className="w-12 h-12 text-ash mx-auto stroke-[1.5]" />
              <h3 className="text-lg font-semibold text-ink">No Job Descriptions Found</h3>
              <p className="text-steel text-sm max-w-md mx-auto">
                No local job descriptions match your search query. Create a new job description to analyze candidate match.
              </p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="btn-primary inline-flex items-center gap-2 text-xs"
              >
                <Plus className="w-4 h-4" />
                <span>New Job Description</span>
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {filteredLocalJobs.map((j) => (
                <div
                  key={j.id}
                  onClick={() => router.push(`/jobs/${j.id}`)}
                  className="card-grafbase p-6 flex flex-col justify-between space-y-5 hover:shadow-md transition-shadow cursor-pointer group"
                >
                  <div className="space-y-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <h3 className="font-semibold text-ink text-lg group-hover:text-teal-700 transition-colors">
                          {j.title}
                        </h3>
                        <p className="text-xs text-steel font-medium flex items-center gap-2 mt-0.5">
                          <span className="flex items-center gap-1"><Building className="w-3.5 h-3.5" /> Local Job Description</span>
                          <span>•</span>
                          <span className="flex items-center gap-1"><MapPin className="w-3.5 h-3.5" /> Remote · Full-time</span>
                        </p>
                      </div>

                      <span className="text-xs font-mono text-steel bg-canvas px-2.5 py-1 rounded border border-hairline shrink-0">
                        Job #{j.id}
                      </span>
                    </div>

                    {/* Requirements Breakdown */}
                    <div className="space-y-2 text-xs">
                      <div>
                        <span className="text-steel font-medium text-[11px] block mb-1">Required:</span>
                        <div className="flex flex-wrap gap-1">
                          {j.required_skills.map((s) => (
                            <span key={s} className="badge-mint px-2 py-0.5 rounded text-[11px]">
                              {s}
                            </span>
                          ))}
                          {j.required_technologies.map((t) => (
                            <span key={t} className="badge-sky px-2 py-0.5 rounded text-[11px]">
                              {t}
                            </span>
                          ))}
                        </div>
                      </div>

                      {(j.preferred_skills.length > 0 || j.preferred_technologies.length > 0) && (
                        <div>
                          <span className="text-steel font-medium text-[11px] block mb-1">Preferred:</span>
                          <div className="flex flex-wrap gap-1">
                            {j.preferred_skills.map((ps) => (
                              <span key={ps} className="badge-moss px-2 py-0.5 rounded text-[11px]">
                                {ps}
                              </span>
                            ))}
                            {j.preferred_technologies.map((pt) => (
                              <span key={pt} className="badge-moss px-2 py-0.5 rounded text-[11px]">
                                {pt}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="pt-4 border-t border-hairline flex items-center justify-between text-xs text-steel">
                    <span className="font-mono">
                      {j.required_skills.length + j.required_technologies.length} Required · {j.preferred_skills.length + j.preferred_technologies.length} Preferred
                    </span>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        router.push(`/jobs/${j.id}`);
                      }}
                      className="btn-primary text-xs py-1.5 px-3 flex items-center gap-1.5"
                    >
                      <span>View Job</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* TAB 2: RECOMMENDED JOBS (Remotive Real Live Listings for Candidate) */}
      {activeTab === "recommended" && (
        <div className="space-y-6">
          {/* Candidate Selection Header Card */}
          <div className="card-grafbase p-6 space-y-4 bg-emerald-50/40 border-emerald-200">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div>
                <span className="text-xs font-mono text-emerald-800 font-semibold bg-emerald-100 px-2.5 py-0.5 rounded border border-emerald-300 inline-block mb-1">
                  Remotive Public API · Real Live Remote Jobs
                </span>
                <h2 className="text-xl font-semibold text-ink">Recommended Remote Opportunities</h2>
                <p className="text-xs text-steel mt-0.5">
                  Extracted requirements from live Remotive job postings scored deterministically against candidate profile.
                </p>
              </div>

              {/* Candidate Dropdown */}
              <div className="flex items-center gap-2 w-full md:w-auto">
                <Users className="w-4 h-4 text-steel shrink-0" />
                <span className="text-xs font-medium text-steel shrink-0">Candidate:</span>
                <select
                  value={selectedCandidateId || ""}
                  onChange={(e) => setSelectedCandidateId(Number(e.target.value))}
                  className="p-2.5 rounded-lg border border-hairline bg-canvas text-xs font-semibold text-ink focus:outline-none focus:ring-2 focus:ring-ink w-full md:w-64"
                >
                  {candidates.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name} ({c.skills.length + c.technologies.length} skills extracted)
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {selectedCandidate && (
              <div className="p-3.5 rounded-xl bg-canvas border border-hairline space-y-2 text-xs">
                <div className="flex items-center justify-between text-steel">
                  <span>Candidate Skills used for Remotive discovery:</span>
                  <button
                    onClick={() => selectedCandidateId && handleFetchRemotive(selectedCandidateId)}
                    disabled={loadingRemotive}
                    className="text-emerald-700 font-semibold hover:underline flex items-center gap-1"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${loadingRemotive ? "animate-spin" : ""}`} />
                    <span>Refresh Remotive Listings</span>
                  </button>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {selectedCandidate.skills.concat(selectedCandidate.technologies).map((st) => (
                    <span key={st} className="badge-sky px-2.5 py-0.5 rounded text-[11px] font-medium">
                      {st}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Results Listing */}
          {loadingRemotive ? (
            <div className="py-16 text-center text-steel font-mono space-y-3">
              <div className="w-6 h-6 border-2 border-steel border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-sm">Fetching live remote job listings from Remotive API for {selectedCandidate?.name}...</p>
            </div>
          ) : filteredRemotiveJobs.length === 0 ? (
            <div className="card-grafbase p-12 text-center text-steel space-y-3">
              <Search className="w-10 h-10 text-ash mx-auto" />
              <p className="font-semibold text-ink text-base">No Matching Remotive Jobs Found</p>
              <p className="text-xs">Try selecting a candidate with more extracted skills or clearing your search term.</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="text-xs text-steel font-mono flex items-center justify-between">
                <span>Showing {filteredRemotiveJobs.length} real remote job listings for {selectedCandidate?.name}:</span>
                <span className="text-ash">Official Remotive API URLs preserved</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {filteredRemotiveJobs.map((rj, idx) => {
                  const pct = Math.round(rj.overall_score * 100);
                  const isHigh = pct >= 75;
                  const isMed = pct >= 50 && pct < 75;

                  return (
                    <div
                      key={rj.remotive_id || idx}
                      className="card-grafbase p-6 flex flex-col justify-between space-y-5 hover:shadow-md transition-shadow cursor-pointer"
                      onClick={() => handleImportRemotiveJob(rj)}
                    >
                      <div className="space-y-4">
                        <div className="flex items-start justify-between">
                          <div>
                            <div className="flex items-center gap-2 text-[11px] font-mono text-steel uppercase mb-1">
                              <span>Remotive Job #{rj.remotive_id}</span>
                              <span>•</span>
                              <span className="text-emerald-700 font-semibold bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                                Live Remotive
                              </span>
                            </div>
                            <h3 className="font-semibold text-ink text-lg">{rj.title}</h3>
                            <p className="text-xs text-steel font-medium flex items-center gap-2 mt-0.5">
                              <span className="flex items-center gap-1"><Building className="w-3.5 h-3.5" /> {rj.company_name}</span>
                              <span>•</span>
                              <span className="flex items-center gap-1"><MapPin className="w-3.5 h-3.5" /> {rj.location || "Remote"}</span>
                            </p>
                          </div>

                          <div
                            className={`px-3 py-1 rounded-full text-xs font-bold font-mono border shrink-0 ${
                              isHigh
                                ? "bg-emerald-50 text-emerald-700 border-emerald-300"
                                : isMed
                                ? "bg-amber-50 text-amber-700 border-amber-300"
                                : "bg-rose-50 text-rose-700 border-rose-300"
                            }`}
                          >
                            {pct}% MATCH
                          </div>
                        </div>

                        {/* Description snippet */}
                        <p className="text-xs text-steel line-clamp-2 italic bg-canvas/40 p-3 rounded-lg border border-hairline font-mono">
                          "{rj.description_snippet}"
                        </p>

                        {/* Matched Skills Breakdown */}
                        <div className="space-y-2 text-xs">
                          <div>
                            <span className="text-steel font-medium text-[11px] block mb-1">Matched Criteria ({rj.matched_skills.length}):</span>
                            <div className="flex flex-wrap gap-1">
                              {rj.matched_skills.length > 0 ? (
                                rj.matched_skills.map((ms) => (
                                  <span key={ms} className="badge-mint px-2 py-0.5 rounded text-[11px]">
                                    ✓ {ms}
                                  </span>
                                ))
                              ) : (
                                <span className="text-ash italic">No direct skill match</span>
                              )}
                            </div>
                          </div>

                          {rj.missing_skills.length > 0 && (
                            <div>
                              <span className="text-steel font-medium text-[11px] block mb-1">Missing Requirements:</span>
                              <div className="flex flex-wrap gap-1">
                                {rj.missing_skills.slice(0, 4).map((mis) => (
                                  <span key={mis} className="bg-rose-50 text-rose-700 border border-rose-200 px-2 py-0.5 rounded text-[11px]">
                                    ✗ {mis}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Footer Actions */}
                      <div className="pt-4 border-t border-hairline flex items-center justify-between text-xs">
                        <a
                          href={rj.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          className="text-xs text-steel hover:text-ink font-semibold flex items-center gap-1 bg-canvas px-2.5 py-1 rounded border border-hairline"
                        >
                          <span>View Official Remotive URL</span>
                          <ExternalLink className="w-3.5 h-3.5" />
                        </a>

                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleImportRemotiveJob(rj);
                          }}
                          disabled={importingId === rj.remotive_id}
                          className="btn-primary text-xs py-1.5 px-3 flex items-center gap-1.5"
                        >
                          <span>{importingId === rj.remotive_id ? "Importing..." : "View Job & Match"}</span>
                          <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* New Job Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="card-grafbase max-w-xl w-full p-6 space-y-5 shadow-2xl animate-in fade-in zoom-in-95">
            <div className="flex items-center justify-between border-b border-hairline pb-3">
              <h3 className="font-semibold text-ink text-base">New Job Description</h3>
              <button onClick={() => setShowCreateModal(false)} className="text-steel hover:text-ink">
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-steel mb-1">Job Title *</label>
                <input
                  type="text"
                  placeholder="e.g. Senior Machine Learning Engineer"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="w-full p-2.5 rounded-lg border border-hairline bg-canvas/30 text-sm focus:outline-none focus:ring-2 focus:ring-ink"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="block text-xs font-medium text-steel">Job Description Text *</label>
                  <button
                    type="button"
                    onClick={fillPreset}
                    className="text-xs text-teal-700 hover:underline font-medium"
                  >
                    Insert Preset Sample JD
                  </button>
                </div>
                <textarea
                  rows={6}
                  placeholder="Paste job description text..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full p-3 rounded-lg border border-hairline bg-canvas/30 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-ink"
                />
              </div>

              {error && <div className="text-xs text-rose-600">{error}</div>}
            </div>

            <div className="flex items-center justify-end gap-3 pt-3 border-t border-hairline">
              <button
                type="button"
                onClick={() => setShowCreateModal(false)}
                className="btn-ghost text-xs py-2 px-4"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleCreateJob}
                disabled={creating}
                className="btn-primary text-xs py-2 px-4"
              >
                {creating ? "Extracting Requirements..." : "Save & Extract Job"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
