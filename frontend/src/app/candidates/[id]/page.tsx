"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  fetchCandidateById,
  recommendRoles,
  fetchJobs,
  CandidateProfile,
  RoleRecommendation,
  JobProfile
} from "@/lib/api";
import {
  User,
  Cpu,
  Layers,
  Globe,
  Award,
  Briefcase,
  ArrowRight,
  RefreshCw,
  ChevronRight,
  CheckCircle,
  XCircle
} from "lucide-react";

export default function CandidateDetailPage() {
  const params = useParams();
  const id = Number(params.id);

  const [candidate, setCandidate] = useState<CandidateProfile | null>(null);
  const [roles, setRoles] = useState<RoleRecommendation[]>([]);
  const [jobs, setJobs] = useState<JobProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [recommending, setRecommending] = useState(false);

  useEffect(() => {
    if (!id) return;

    async function loadData() {
      setLoading(true);
      try {
        const cand = await fetchCandidateById(id);
        setCandidate(cand);
        const jList = await fetchJobs();
        setJobs(jList);

        // Fetch initial role recommendations
        const recs = await recommendRoles(id, 5);
        setRoles(recs);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [id]);

  const handleRecommendRoles = async () => {
    if (!candidate) return;
    setRecommending(true);
    try {
      const recs = await recommendRoles(candidate.id, 5);
      setRoles(recs);
    } catch (err) {
      console.error(err);
    } finally {
      setRecommending(false);
    }
  };

  if (loading) {
    return <div className="py-20 text-center text-steel font-mono">Loading candidate profile...</div>;
  }

  if (!candidate) {
    return <div className="py-20 text-center text-rose-600">Candidate profile not found.</div>;
  }

  return (
    <div className="space-y-8">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-xs text-steel font-mono">
        <Link href="/candidates" className="hover:text-ink">Candidates</Link>
        <ChevronRight className="w-3.5 h-3.5" />
        <span className="text-ink font-medium">{candidate.name}</span>
      </div>

      {/* Profile Header Card */}
      <div className="card-grafbase p-8 space-y-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-hairline pb-6">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <User className="w-5 h-5 text-ink" />
              <h1 className="text-2xl md:text-3xl font-semibold text-ink">{candidate.name}</h1>
              <span className="text-xs font-mono text-steel bg-canvas px-2.5 py-1 rounded-full border border-hairline">
                Candidate #{candidate.id}
              </span>
            </div>
            {candidate.contact_info && (
              <p className="text-sm text-steel">{candidate.contact_info}</p>
            )}
          </div>

          <button
            onClick={handleRecommendRoles}
            disabled={recommending}
            className="btn-primary flex items-center gap-2 text-xs"
          >
            <RefreshCw className={`w-4 h-4 ${recommending ? "animate-spin" : ""}`} />
            <span>{recommending ? "Calculating Fit..." : "Refresh Role Recommendations"}</span>
          </button>
        </div>

        {/* Extracted Profile Tags */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Skills */}
          <div className="p-4 rounded-xl bg-canvas/40 border border-hairline space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-steel">
              <Cpu className="w-4 h-4 text-teal-600" />
              <span>Skills ({candidate.skills.length})</span>
            </div>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {candidate.skills.length > 0 ? (
                candidate.skills.map((s) => (
                  <span key={s} className="badge-mint px-2.5 py-1 rounded text-xs font-medium">
                    {s}
                  </span>
                ))
              ) : (
                <span className="text-xs text-ash italic">None detected</span>
              )}
            </div>
          </div>

          {/* Technologies */}
          <div className="p-4 rounded-xl bg-canvas/40 border border-hairline space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-steel">
              <Layers className="w-4 h-4 text-sky-600" />
              <span>Technologies ({candidate.technologies.length})</span>
            </div>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {candidate.technologies.length > 0 ? (
                candidate.technologies.map((t) => (
                  <span key={t} className="badge-sky px-2.5 py-1 rounded text-xs font-medium">
                    {t}
                  </span>
                ))
              ) : (
                <span className="text-xs text-ash italic">None detected</span>
              )}
            </div>
          </div>

          {/* Languages */}
          <div className="p-4 rounded-xl bg-canvas/40 border border-hairline space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-steel">
              <Globe className="w-4 h-4 text-lime-600" />
              <span>Languages ({candidate.languages.length})</span>
            </div>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {candidate.languages.length > 0 ? (
                candidate.languages.map((l) => (
                  <span key={l} className="badge-moss px-2.5 py-1 rounded text-xs font-medium">
                    {l}
                  </span>
                ))
              ) : (
                <span className="text-xs text-ash italic">None detected</span>
              )}
            </div>
          </div>
        </div>

        {/* Raw Description Text */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-steel uppercase tracking-wider">Raw Input Description</h4>
          <p className="text-xs text-ink bg-canvas/60 p-4 rounded-xl border border-hairline font-mono leading-relaxed">
            {candidate.raw_text}
          </p>
        </div>
      </div>

      {/* Part 2 Section — Job Role Recommendations */}
      <div className="space-y-6">
        <div className="flex items-center justify-between border-b border-hairline pb-4">
          <div>
            <div className="text-xs font-mono text-steel uppercase tracking-wider">Part 2 — Recommendation Engine</div>
            <h2 className="text-2xl font-semibold text-ink">Top Job Role Recommendations</h2>
          </div>
          <span className="text-xs text-steel font-mono">Ranked by Fit Score</span>
        </div>

        {roles.length === 0 ? (
          <div className="card-grafbase p-8 text-center text-steel">No role recommendations generated yet.</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {roles.map((r, idx) => {
              const pct = Math.round(r.score * 100);
              const isHigh = pct >= 75;
              const isMed = pct >= 50 && pct < 75;

              return (
                <div key={r.id || idx} className="card-grafbase p-6 flex flex-col justify-between space-y-5">
                  <div className="space-y-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <span className="text-[11px] font-mono text-steel uppercase">Rank #{idx + 1}</span>
                        <h3 className="font-semibold text-ink text-base mt-0.5">{r.role_name}</h3>
                      </div>

                      {/* Score Badge */}
                      <div
                        className={`px-3 py-1 rounded-full text-xs font-bold font-mono border ${
                          isHigh
                            ? "bg-emerald-50 text-emerald-700 border-emerald-300"
                            : isMed
                            ? "bg-amber-50 text-amber-700 border-amber-300"
                            : "bg-rose-50 text-rose-700 border-rose-300"
                        }`}
                      >
                        {pct}% Fit
                      </div>
                    </div>

                    {/* Why Recommended Rationale Box */}
                    {r.score_reasons && r.score_reasons.length > 0 && (
                      <div className="p-2.5 rounded-lg bg-canvas/60 border border-hairline space-y-1">
                        <span className="text-[10px] text-steel font-mono font-semibold uppercase block">Why Recommended:</span>
                        <p className="text-[11px] text-ink italic leading-tight">
                          "{r.score_reasons[0]}"
                        </p>
                      </div>
                    )}

                    {/* Matched & Missing Skills */}
                    <div className="space-y-3 text-xs">
                      <div>
                        <span className="text-steel font-medium flex items-center gap-1 mb-1">
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                          <span>Matched Skills ({r.matched_skills.length})</span>
                        </span>
                        <div className="flex flex-wrap gap-1">
                          {r.matched_skills.length > 0 ? (
                            r.matched_skills.map((ms) => (
                              <span key={ms} className="badge-mint px-2 py-0.5 rounded text-[11px]">
                                {ms}
                              </span>
                            ))
                          ) : (
                            <span className="text-ash italic">No direct skill overlap</span>
                          )}
                        </div>
                      </div>

                      {r.missing_skills.length > 0 && (
                        <div>
                          <span className="text-steel font-medium flex items-center gap-1 mb-1">
                            <XCircle className="w-3.5 h-3.5 text-rose-500" />
                            <span>Missing Skill Gaps</span>
                          </span>
                          <div className="flex flex-wrap gap-1">
                            {r.missing_skills.map((mis) => (
                              <span key={mis} className="bg-rose-50 text-rose-700 border border-rose-200 px-2 py-0.5 rounded text-[11px]">
                                {mis}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="pt-3 border-t border-hairline space-y-1.5 font-mono text-[11px] text-steel">
                    <div className="flex justify-between items-center">
                      <span>Skill Score:</span>
                      <span className="font-semibold text-ink">{Math.round(r.skill_score * 100)}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span>Technology Score:</span>
                      <span className="font-semibold text-ink">{Math.round(r.tech_score * 100)}%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span>Semantic Score:</span>
                      <span className="font-semibold text-ink">{Math.round((r.semantic_score || 0) * 100)}%</span>
                    </div>
                    <div className="flex justify-between items-center pt-1 border-t border-hairline/60 text-ink font-semibold">
                      <span>Overall Fit:</span>
                      <span className="text-emerald-700 font-bold">{pct}%</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Match against existing job profiles */}
      {jobs.length > 0 && (
        <div className="card-grafbase p-6 space-y-4">
          <h3 className="font-semibold text-ink text-lg flex items-center gap-2">
            <Briefcase className="w-5 h-5 text-steel" />
            <span>Match {candidate.name} Against Active Job Descriptions</span>
          </h3>
          <p className="text-xs text-steel">
            Select an active job from your dataset to view a full deterministic score breakdown.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 pt-2">
            {jobs.map((j) => (
              <Link
                key={j.id}
                href={`/match/${candidate.id}/${j.id}`}
                className="p-4 rounded-xl border border-hairline bg-canvas/40 hover:bg-marble hover:shadow-sm transition-all flex items-center justify-between group"
              >
                <div>
                  <h4 className="font-semibold text-ink text-sm group-hover:text-teal-700">{j.title}</h4>
                  <span className="text-xs text-steel font-mono">Job #{j.id}</span>
                </div>
                <ArrowRight className="w-4 h-4 text-steel group-hover:translate-x-1 transition-transform" />
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
