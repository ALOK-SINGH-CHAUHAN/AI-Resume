"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { fetchJobById, fetchCandidateRankingsForJob, JobProfile } from "@/lib/api";
import {
  Briefcase,
  ChevronRight,
  ShieldCheck,
  Users,
  CheckCircle2,
  XCircle,
  PlusCircle,
  ArrowRight,
  AlertTriangle,
  GitMerge
} from "lucide-react";

export default function JobDetailPage() {
  const params = useParams();
  const id = Number(params.id);

  const [job, setJob] = useState<JobProfile | null>(null);
  const [rankings, setRankings] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;

    async function loadData() {
      setLoading(true);
      try {
        const j = await fetchJobById(id);
        setJob(j);
        const rData = await fetchCandidateRankingsForJob(id);
        setRankings(rData.rankings);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [id]);

  if (loading) {
    return (
      <div className="py-20 text-center text-steel">
        <div className="w-6 h-6 border-2 border-steel border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="font-mono text-sm">Loading job & candidate rankings...</p>
      </div>
    );
  }

  if (!job) {
    return <div className="py-20 text-center text-rose-600">Job description not found.</div>;
  }

  return (
    <div className="space-y-8">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-xs text-steel font-mono">
        <Link href="/jobs" className="hover:text-ink">Jobs</Link>
        <ChevronRight className="w-3.5 h-3.5" />
        <span className="text-ink font-medium">{job.title}</span>
      </div>

      {/* Job Info Header Card */}
      <div className="card-grafbase p-8 space-y-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-hairline pb-6">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Briefcase className="w-5 h-5 text-ink" />
              <h1 className="text-2xl md:text-3xl font-semibold text-ink">{job.title}</h1>
              <span className="text-xs font-mono text-steel bg-canvas px-2.5 py-1 rounded-full border border-hairline">
                Job #{job.id}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono text-emerald-700 bg-emerald-50 px-3 py-1.5 rounded-full border border-emerald-200">
            <ShieldCheck className="w-4 h-4" />
            <span>Deterministic Ranking Active</span>
          </div>
        </div>

        {/* Required & Preferred Criteria Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 rounded-xl bg-canvas/40 border border-hairline space-y-2">
            <span className="text-xs font-semibold text-steel uppercase tracking-wider block">
              Required Criteria ({job.required_skills.length + job.required_technologies.length})
            </span>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {job.required_skills.map((s) => (
                <span key={s} className="badge-mint px-2.5 py-1 rounded text-xs font-medium">
                  {s}
                </span>
              ))}
              {job.required_technologies.map((t) => (
                <span key={t} className="badge-sky px-2.5 py-1 rounded text-xs font-medium">
                  {t}
                </span>
              ))}
            </div>
          </div>

          <div className="p-4 rounded-xl bg-canvas/40 border border-hairline space-y-2">
            <span className="text-xs font-semibold text-steel uppercase tracking-wider block">
              Preferred / Bonus Criteria ({job.preferred_skills.length + job.preferred_technologies.length})
            </span>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {job.preferred_skills.concat(job.preferred_technologies).length > 0 ? (
                job.preferred_skills.concat(job.preferred_technologies).map((p) => (
                  <span key={p} className="badge-moss px-2.5 py-1 rounded text-xs font-medium">
                    {p}
                  </span>
                ))
              ) : (
                <span className="text-xs text-ash italic">No preferred criteria specified</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Candidate Fit Leaderboard */}
      <div className="space-y-6">
        <div className="flex items-center justify-between border-b border-hairline pb-4">
          <div>
            <div className="text-xs font-mono text-steel uppercase tracking-wider">Part 2 · Candidate Ranking</div>
            <h2 className="text-2xl font-semibold text-ink">Deterministic Fit Leaderboard</h2>
          </div>
          <span className="text-xs text-steel font-mono bg-canvas px-3 py-1 rounded-full border border-hairline">
            {rankings.length} Candidate(s) Ranked
          </span>
        </div>

        {rankings.length === 0 ? (
          <div className="card-grafbase p-12 text-center space-y-4">
            <Users className="w-12 h-12 text-ash mx-auto stroke-[1.5]" />
            <h3 className="text-lg font-semibold text-ink">No Candidates Saved to Rank</h3>
            <p className="text-steel text-sm max-w-md mx-auto">
              Extract candidate self-descriptions to rank them against this job description.
            </p>
            <Link href="/extract" className="btn-primary inline-flex items-center gap-2 text-xs">
              <span>Go to Candidate Extraction</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {rankings.map((r, idx) => {
              const pct = Math.round(r.overall_score * 100);
              const isHigh = pct >= 75;
              const isMed = pct >= 50 && pct < 75;
              const hardGaps: string[] = r.hard_gaps || [];
              const hasHardGaps: boolean = r.has_hard_gaps || hardGaps.length > 0;
              const relatedCompetencies: string[] = r.related_competencies || [];
              const matchedRequired: string[] = r.matched_required || r.matched_skills || [];
              const matchedPreferred: string[] = r.matched_preferred || [];

              return (
                <div key={r.candidate_id} className="card-grafbase p-6 space-y-4 hover:shadow-md transition-shadow">
                  {/* Hard gap alert for this candidate */}
                  {hasHardGaps && (
                    <div className="flex items-center gap-2 p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700">
                      <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                      <span>
                        Hard gap: missing required criteria — <strong>{hardGaps.join(", ")}</strong>
                      </span>
                    </div>
                  )}

                  <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-hairline pb-4">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-xl bg-canvas border border-hairline flex items-center justify-center font-mono font-bold text-ink text-sm">
                        {String(idx + 1).padStart(2, "0")}
                      </div>

                      <div>
                        <h3 className="font-semibold text-ink text-lg">{r.candidate_name}</h3>
                        <div className="flex items-center gap-3 text-xs text-steel font-mono mt-0.5">
                          <span>Skill: {Math.round(r.skill_score * 100)}%</span>
                          <span>•</span>
                          <span>Tech: {Math.round(r.tech_score * 100)}%</span>
                          <span>•</span>
                          <span>Semantic: {Math.round(r.semantic_score * 100)}%</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <div
                        className={`px-4 py-2 rounded-xl text-sm font-bold font-mono border ${
                          isHigh
                            ? "bg-emerald-50 text-emerald-700 border-emerald-300"
                            : isMed
                            ? "bg-amber-50 text-amber-700 border-amber-300"
                            : "bg-rose-50 text-rose-700 border-rose-300"
                        }`}
                      >
                        {pct}% Fit
                      </div>

                      <Link
                        href={`/match/${r.candidate_id}/${job.id}`}
                        className="btn-ghost text-xs py-2 px-3 flex items-center gap-1.5"
                      >
                        <span>Inspect Match</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </Link>
                    </div>
                  </div>

                  {/* Match breakdown: Required / Preferred / Related / Gaps */}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
                    {/* Required Met */}
                    <div className="space-y-1.5 p-3 rounded-xl bg-canvas/30 border border-hairline">
                      <span className="text-steel font-medium flex items-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                        <span>Required Met ({matchedRequired.length})</span>
                      </span>
                      <div className="flex flex-wrap gap-1 pt-0.5">
                        {matchedRequired.length > 0 ? (
                          matchedRequired.slice(0, 3).map((ms: string) => (
                            <span key={ms} className="badge-mint px-2 py-0.5 rounded text-[11px] font-medium">
                              ✓ {ms}
                            </span>
                          ))
                        ) : (
                          <span className="text-ash italic">None</span>
                        )}
                        {matchedRequired.length > 3 && (
                          <span className="text-steel text-[11px]">+{matchedRequired.length - 3}</span>
                        )}
                      </div>
                    </div>

                    {/* Preferred Met */}
                    <div className="space-y-1.5 p-3 rounded-xl bg-canvas/30 border border-hairline">
                      <span className="text-steel font-medium flex items-center gap-1">
                        <span className="w-3.5 h-3.5 text-amber-500 font-bold">★</span>
                        <span>Preferred Met ({matchedPreferred.length})</span>
                      </span>
                      <div className="flex flex-wrap gap-1 pt-0.5">
                        {matchedPreferred.length > 0 ? (
                          matchedPreferred.slice(0, 3).map((ps: string) => (
                            <span key={ps} className="badge-moss px-2 py-0.5 rounded text-[11px]">
                              ✓ {ps}
                            </span>
                          ))
                        ) : (
                          <span className="text-ash italic">None</span>
                        )}
                      </div>
                    </div>

                    {/* Related Competencies */}
                    <div className="space-y-1.5 p-3 rounded-xl bg-canvas/30 border border-hairline">
                      <span className="text-steel font-medium flex items-center gap-1">
                        <GitMerge className="w-3.5 h-3.5 text-indigo-500" />
                        <span>Related ({relatedCompetencies.length})</span>
                      </span>
                      <div className="flex flex-wrap gap-1 pt-0.5">
                        {relatedCompetencies.length > 0 ? (
                          relatedCompetencies.slice(0, 2).map((rc: string) => (
                            <span key={rc} className="bg-indigo-50 text-indigo-700 border border-indigo-200 px-2 py-0.5 rounded text-[11px]">
                              ~ {rc}
                            </span>
                          ))
                        ) : (
                          <span className="text-ash italic">None</span>
                        )}
                      </div>
                    </div>

                    {/* Hard Gaps */}
                    <div className="space-y-1.5 p-3 rounded-xl bg-canvas/30 border border-hairline">
                      <span className="text-steel font-medium flex items-center gap-1">
                        <XCircle className="w-3.5 h-3.5 text-rose-600" />
                        <span>Hard Gaps ({hardGaps.length})</span>
                      </span>
                      <div className="flex flex-wrap gap-1 pt-0.5">
                        {hardGaps.length > 0 ? (
                          hardGaps.slice(0, 3).map((mis: string) => (
                            <span key={mis} className="bg-rose-50 text-rose-700 border border-rose-200 px-2 py-0.5 rounded text-[11px]">
                              ✗ {mis}
                            </span>
                          ))
                        ) : (
                          <span className="text-emerald-700 font-medium">✓ Zero gaps</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
