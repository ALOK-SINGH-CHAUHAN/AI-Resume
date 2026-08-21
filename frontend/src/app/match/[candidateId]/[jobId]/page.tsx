"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { matchCandidateAndJob, fetchCandidateById, fetchJobById, MatchResult, CandidateProfile, JobProfile } from "@/lib/api";
import {
  ChevronRight,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  PlusCircle,
  Cpu,
  Layers,
  Info,
  Quote,
  Sliders,
  Check,
  AlertTriangle,
  Star,
  Award,
  GitMerge
} from "lucide-react";

export default function MatchDetailPage() {
  const params = useParams();
  const candidateId = Number(params.candidateId);
  const jobId = Number(params.jobId);

  const [matchResult, setMatchResult] = useState<MatchResult | null>(null);
  const [candidate, setCandidate] = useState<CandidateProfile | null>(null);
  const [job, setJob] = useState<JobProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!candidateId || !jobId) return;

    async function loadData() {
      setLoading(true);
      try {
        const cand = await fetchCandidateById(candidateId);
        setCandidate(cand);
        const j = await fetchJobById(jobId);
        setJob(j);
        const m = await matchCandidateAndJob(candidateId, jobId);
        setMatchResult(m);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [candidateId, jobId]);

  if (loading) {
    return (
      <div className="py-20 text-center text-steel">
        <div className="w-6 h-6 border-2 border-steel border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="font-mono text-sm">Computing deterministic match score...</p>
      </div>
    );
  }

  if (!matchResult || !candidate || !job) {
    return <div className="py-20 text-center text-rose-600">Failed to load match details.</div>;
  }

  const overallPct = Math.round(matchResult.overall_score * 100);
  const skillPct = Math.round(matchResult.skill_score * 100);
  const techPct = Math.round(matchResult.tech_score * 100);
  const semanticPct = Math.round(matchResult.semantic_score * 100);

  const isHigh = overallPct >= 75;
  const isMed = overallPct >= 50 && overallPct < 75;

  const hardGaps = matchResult.hard_gaps || [];
  const hasHardGaps = matchResult.has_hard_gaps || hardGaps.length > 0;
  const matchedRequired = matchResult.matched_required || [];
  const matchedPreferred = matchResult.matched_preferred || [];
  const relatedCompetencies = matchResult.related_competencies || [];
  const missingPreferred = matchResult.missing_preferred || [];

  return (
    <div className="space-y-8">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-xs text-steel font-mono">
        <Link href="/candidates" className="hover:text-ink">Candidates</Link>
        <ChevronRight className="w-3.5 h-3.5" />
        <Link href={`/candidates/${candidate.id}`} className="hover:text-ink">{candidate.name}</Link>
        <ChevronRight className="w-3.5 h-3.5" />
        <span className="text-ink font-medium">Match against {job.title}</span>
      </div>

      {/* Hard Gap Alert Banner */}
      {hasHardGaps && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="text-sm font-semibold text-rose-800">Hard Gap Warning</p>
            <p className="text-xs text-rose-700">
              Candidate is missing {hardGaps.length} required criteria with no related competency:
              {" "}<strong>{hardGaps.join(", ")}</strong>.
              This may be a significant fit risk.
            </p>
          </div>
        </div>
      )}

      {/* Match Banner Card */}
      <div className="card-grafbase p-8 space-y-6">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6 border-b border-hairline pb-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-mono text-emerald-700 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200 w-fit">
              <ShieldCheck className="w-4 h-4" />
              <span>Deterministic Match — Rule-Based Formula</span>
            </div>
            <h1 className="text-2xl md:text-3xl font-semibold text-ink">
              {candidate.name} <span className="text-steel font-normal">vs</span> {job.title}
            </h1>
            <p className="text-xs text-steel font-mono">
              Score formula: 45% Skill Match · 30% Technology Match · 25% Semantic Similarity
            </p>
          </div>

          <div className="flex items-center gap-4 bg-canvas p-4 rounded-2xl border border-hairline w-full lg:w-auto justify-between lg:justify-start">
            <div className="text-right">
              <span className="text-[11px] font-mono text-steel uppercase block">Overall Match</span>
              <span className="text-xs text-emerald-700 font-mono font-semibold">Deterministic Result</span>
            </div>
            <div
              className={`text-3xl font-bold font-mono px-5 py-3 rounded-xl border ${
                isHigh
                  ? "bg-emerald-50 text-emerald-700 border-emerald-300 shadow-sm"
                  : isMed
                  ? "bg-amber-50 text-amber-700 border-amber-300"
                  : "bg-rose-50 text-rose-700 border-rose-300"
              }`}
            >
              {overallPct}%
            </div>
          </div>
        </div>

        {/* Subscore Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-5 rounded-2xl bg-canvas/40 border border-hairline space-y-2">
            <div className="flex justify-between items-center text-xs font-medium text-steel">
              <span className="flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-teal-600" />
                <span>Skill Match</span>
              </span>
              <span className="font-mono font-bold text-ink">{skillPct}%</span>
            </div>
            <div className="w-full bg-zinc-200 h-2 rounded-full overflow-hidden">
              <div className="bg-teal-600 h-full rounded-full transition-all" style={{ width: `${skillPct}%` }} />
            </div>
            <div className="flex justify-between text-[11px] font-mono text-steel pt-1 border-t border-hairline/60">
              <span>Weight: 45%</span>
              <span className="font-semibold text-teal-700">+{(matchResult.skill_score * 45).toFixed(2)}%</span>
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-canvas/40 border border-hairline space-y-2">
            <div className="flex justify-between items-center text-xs font-medium text-steel">
              <span className="flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-sky-600" />
                <span>Technology Match</span>
              </span>
              <span className="font-mono font-bold text-ink">{techPct}%</span>
            </div>
            <div className="w-full bg-zinc-200 h-2 rounded-full overflow-hidden">
              <div className="bg-sky-600 h-full rounded-full transition-all" style={{ width: `${techPct}%` }} />
            </div>
            <div className="flex justify-between text-[11px] font-mono text-steel pt-1 border-t border-hairline/60">
              <span>Weight: 30%</span>
              <span className="font-semibold text-sky-700">+{(matchResult.tech_score * 30).toFixed(2)}%</span>
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-canvas/40 border border-hairline space-y-2">
            <div className="flex justify-between items-center text-xs font-medium text-steel">
              <span className="flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-indigo-600" />
                <span>Semantic Match</span>
              </span>
              <span className="font-mono font-bold text-ink">{semanticPct}%</span>
            </div>
            <div className="w-full bg-zinc-200 h-2 rounded-full overflow-hidden">
              <div className="bg-indigo-600 h-full rounded-full transition-all" style={{ width: `${semanticPct}%` }} />
            </div>
            <div className="flex justify-between text-[11px] font-mono text-steel pt-1 border-t border-hairline/60">
              <span>Weight: 25%</span>
              <span className="font-semibold text-indigo-700">+{(matchResult.semantic_score * 25).toFixed(2)}%</span>
            </div>
          </div>
        </div>

        {/* Explicit Weighted Derivation Summary Table */}
        <div className="p-4 rounded-xl bg-canvas border border-hairline space-y-3 font-mono text-xs">
          <div className="flex items-center justify-between text-steel font-semibold border-b border-hairline pb-2">
            <span>Score Component Derivation</span>
            <span>Raw Score × Formula Weight = Weighted Contribution</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-steel">
            <div className="p-2.5 rounded-lg bg-marble border border-hairline flex justify-between items-center">
              <span>Skill Match:</span>
              <span>{skillPct}% × 45% = <strong className="text-ink">{(matchResult.skill_score * 45).toFixed(2)} pts</strong></span>
            </div>
            <div className="p-2.5 rounded-lg bg-marble border border-hairline flex justify-between items-center">
              <span>Technology Match:</span>
              <span>{techPct}% × 30% = <strong className="text-ink">{(matchResult.tech_score * 30).toFixed(2)} pts</strong></span>
            </div>
            <div className="p-2.5 rounded-lg bg-marble border border-hairline flex justify-between items-center">
              <span>Semantic Match:</span>
              <span>{semanticPct}% × 25% = <strong className="text-ink">{(matchResult.semantic_score * 25).toFixed(2)} pts</strong></span>
            </div>
          </div>
          <div className="flex items-center justify-between pt-2 border-t border-hairline text-ink font-semibold">
            <span>Deterministic Score Sum:</span>
            <span className="text-sm font-bold text-emerald-700">
              {(matchResult.skill_score * 45).toFixed(2)} + {(matchResult.tech_score * 30).toFixed(2)} + {(matchResult.semantic_score * 25).toFixed(2)} = {(matchResult.overall_score * 100).toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      {/* Score Rationale */}
      {matchResult.score_reasons && matchResult.score_reasons.length > 0 && (
        <div className="card-grafbase p-6 space-y-4">
          <h3 className="font-semibold text-ink text-base flex items-center gap-2 border-b border-hairline pb-3">
            <Info className="w-4 h-4 text-steel" />
            <span>Score Rationale</span>
          </h3>
          <ul className="space-y-2">
            {matchResult.score_reasons.map((reason, idx) => (
              <li key={idx} className="flex items-start gap-2 text-sm text-ink">
                <span className="text-emerald-600 font-bold mt-0.5">·</span>
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Requirement Classification Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Matched Required */}
        <div className="card-grafbase p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-hairline pb-3">
            <h3 className="font-semibold text-ink text-base flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              <span>Required Criteria Met ({matchedRequired.length})</span>
            </h3>
            <span className="text-xs font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
              Must Have
            </span>
          </div>
          {matchedRequired.length > 0 ? (
            <div className="flex flex-wrap gap-2 pt-1">
              {matchedRequired.map((s) => (
                <span key={s} className="badge-mint px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5">
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                  <span>{s}</span>
                </span>
              ))}
            </div>
          ) : (
            <div className="py-4 text-center text-steel text-xs italic">No required skills matched.</div>
          )}
        </div>

        {/* Hard Gaps — missing required with no ontology neighbor */}
        <div className="card-grafbase p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-hairline pb-3">
            <h3 className="font-semibold text-ink text-base flex items-center gap-2">
              <XCircle className="w-5 h-5 text-rose-600" />
              <span>Hard Gaps — Required Not Met ({hardGaps.length})</span>
            </h3>
            <span className="text-xs font-mono text-rose-700 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
              Gap Analysis
            </span>
          </div>
          {hardGaps.length > 0 ? (
            <div className="flex flex-wrap gap-2 pt-1">
              {hardGaps.map((s) => (
                <span key={s} className="bg-rose-50 text-rose-700 border border-rose-200 px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5">
                  <XCircle className="w-3.5 h-3.5 text-rose-500" />
                  <span>{s}</span>
                </span>
              ))}
            </div>
          ) : (
            <div className="py-4 text-center text-emerald-700 text-xs font-medium">
              ✓ Zero hard gaps — candidate meets all required criteria.
            </div>
          )}
        </div>
      </div>

      {/* Preferred + Related Competencies */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Preferred Matches */}
        <div className="card-grafbase p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-hairline pb-3">
            <h3 className="font-semibold text-ink text-sm flex items-center gap-2">
              <Star className="w-4 h-4 text-amber-500" />
              <span>Preferred Criteria Met ({matchedPreferred.length})</span>
            </h3>
            <span className="text-xs font-mono text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
              Nice To Have
            </span>
          </div>
          {matchedPreferred.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {matchedPreferred.map((s) => (
                <span key={s} className="badge-moss px-2.5 py-1 rounded text-xs font-medium">
                  ✓ {s}
                </span>
              ))}
            </div>
          ) : (
            <div className="py-3 text-center text-ash text-xs italic">No preferred criteria matched.</div>
          )}
          {missingPreferred.length > 0 && (
            <div className="pt-2 border-t border-hairline space-y-1.5">
              <span className="text-[11px] text-steel font-medium">Missing Preferred:</span>
              <div className="flex flex-wrap gap-1">
                {missingPreferred.map((s) => (
                  <span key={s} className="bg-zinc-100 text-zinc-600 border border-zinc-200 px-2 py-0.5 rounded text-[11px]">
                    ○ {s}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Related Competencies via Ontology */}
        <div className="card-grafbase p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-hairline pb-3">
            <h3 className="font-semibold text-ink text-sm flex items-center gap-2">
              <GitMerge className="w-4 h-4 text-indigo-500" />
              <span>Related Competencies ({relatedCompetencies.length})</span>
            </h3>
            <span className="text-xs font-mono text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200">
              Ontology Detected
            </span>
          </div>
          {relatedCompetencies.length > 0 ? (
            <>
              <p className="text-xs text-steel">
                Candidate has parent or child skills for these job requirements — partial expertise detected.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {relatedCompetencies.map((s) => (
                  <span key={s} className="bg-indigo-50 text-indigo-700 border border-indigo-200 px-2.5 py-1 rounded text-xs">
                    ~ {s}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <div className="py-3 text-center text-ash text-xs italic">No related competency overlaps detected.</div>
          )}
        </div>
      </div>

      {/* Evidence Panel */}
      {matchResult.evidence && matchResult.evidence.length > 0 && (
        <div className="card-grafbase p-6 space-y-4">
          <h3 className="font-semibold text-ink text-base flex items-center gap-2 border-b border-hairline pb-3">
            <Quote className="w-4 h-4 text-steel" />
            <span>Match Evidence — Candidate vs Job Requirement Quotes</span>
          </h3>

          <div className="space-y-3">
            {matchResult.evidence.map((ev, idx) => (
              <div key={idx} className="p-4 rounded-xl bg-canvas/40 border border-hairline grid grid-cols-1 md:grid-cols-12 gap-4 text-xs">
                <div className="md:col-span-3 space-y-1">
                  <span className="font-bold text-ink font-mono text-sm">{ev.term}</span>
                  <span className="block text-[11px] text-emerald-700 font-mono font-medium">{ev.relationship}</span>
                </div>

                <div className="md:col-span-4 bg-marble p-2.5 rounded-lg border border-hairline space-y-0.5">
                  <span className="text-[10px] text-steel font-mono uppercase">Candidate Quote</span>
                  <p className="text-ink font-mono text-[11px] italic">"{ev.candidate_snippet}"</p>
                </div>

                <div className="md:col-span-5 bg-marble p-2.5 rounded-lg border border-hairline space-y-0.5">
                  <span className="text-[10px] text-steel font-mono uppercase">Job Requirement Quote</span>
                  <p className="text-ink font-mono text-[11px] italic">"{ev.job_snippet}"</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Extra Strengths */}
      {matchResult.extra_skills && matchResult.extra_skills.length > 0 && (
        <div className="card-grafbase p-6 space-y-3">
          <h3 className="font-semibold text-ink text-sm flex items-center gap-2">
            <Award className="w-4 h-4 text-sky-600" />
            <span>Additional Candidate Strengths (Beyond JD Scope)</span>
          </h3>
          <p className="text-xs text-steel">Skills and tech the candidate has that were not in the job description.</p>
          <div className="flex flex-wrap gap-1.5">
            {matchResult.extra_skills.map((es) => (
              <span key={es} className="badge-sky px-2.5 py-1 rounded text-xs">
                + {es}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
