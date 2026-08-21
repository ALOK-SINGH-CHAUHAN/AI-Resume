"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchCandidates, deleteCandidate, CandidateProfile } from "@/lib/api";
import { Users, UserPlus, ArrowRight, Trash2, Cpu, Layers, Globe, FileText, Calendar, ShieldCheck } from "lucide-react";

export default function CandidatesPage() {
  const [candidates, setCandidates] = useState<CandidateProfile[]>([]);
  const [loading, setLoading] = useState(true);

  const loadCandidates = async () => {
    setLoading(true);
    try {
      const data = await fetchCandidates();
      setCandidates(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCandidates();
  }, []);

  const handleDelete = async (id: number, name: string) => {
    if (!confirm(`Are you sure you want to delete profile for "${name}"?`)) return;
    try {
      await deleteCandidate(id);
      loadCandidates();
    } catch (err: any) {
      alert(err.message || "Failed to delete candidate");
    }
  };

  const formatDate = (isoStr?: string) => {
    if (!isoStr) return "Recently";
    try {
      return new Date(isoStr).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric"
      });
    } catch {
      return "Recently";
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-hairline pb-6">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-steel uppercase tracking-wider mb-1">
            Part 1 · Candidate Database
          </div>
          <h1 className="text-3xl font-semibold text-ink tracking-tight">Candidate Profiles</h1>
          <p className="text-sm text-steel mt-1">
            Deduplicated candidate database stored locally with deterministic NLP skill extractions.
          </p>
        </div>

        <Link href="/extract" className="btn-primary flex items-center gap-2 text-xs">
          <UserPlus className="w-4 h-4" />
          <span>New Candidate Extraction</span>
        </Link>
      </div>

      {loading ? (
        <div className="text-center py-16 text-steel font-mono text-sm">Loading candidates...</div>
      ) : candidates.length === 0 ? (
        <div className="card-grafbase p-12 text-center space-y-4">
          <Users className="w-12 h-12 text-ash mx-auto stroke-[1.5]" />
          <h3 className="text-lg font-semibold text-ink">No Candidates Saved Yet</h3>
          <p className="text-steel text-sm max-w-md mx-auto">
            Extract entities from conversational descriptions or resume PDFs to populate your candidate database.
          </p>
          <Link href="/extract" className="btn-primary inline-flex items-center gap-2 text-xs">
            <span>Go to Extraction Tool</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {candidates.map((cand) => {
            const totalEntities = cand.skills.length + cand.technologies.length + cand.languages.length;
            const isPdf = cand.raw_text.length > 500 || cand.raw_text.toLowerCase().includes("experience");

            return (
              <div key={cand.id} className="card-grafbase p-6 flex flex-col justify-between space-y-5 hover:shadow-md transition-shadow">
                <div className="space-y-4">
                  {/* Top Name & ID Header */}
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-mono text-steel uppercase bg-canvas px-2 py-0.5 rounded border border-hairline">
                          Candidate #{cand.id}
                        </span>
                        <span className="text-[11px] font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 flex items-center gap-1">
                          <ShieldCheck className="w-3 h-3" />
                          <span>Deduplicated</span>
                        </span>
                      </div>
                      <h3 className="font-semibold text-ink text-xl mt-1">{cand.name}</h3>
                      {cand.contact_info && (
                        <p className="text-xs text-steel">{cand.contact_info}</p>
                      )}
                    </div>

                    <button
                      onClick={() => handleDelete(cand.id, cand.name)}
                      className="text-ash hover:text-rose-600 p-1.5 rounded-lg hover:bg-rose-50 transition-colors"
                      title="Delete Candidate"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Metadata Row */}
                  <div className="flex items-center justify-between text-xs text-steel font-mono bg-canvas/40 p-2.5 rounded-lg border border-hairline">
                    <span className="flex items-center gap-1">
                      <FileText className="w-3.5 h-3.5 text-steel" />
                      <span>{isPdf ? "PDF Document" : "Direct Text"}</span>
                    </span>
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3.5 h-3.5 text-steel" />
                      <span>{formatDate(cand.created_at)}</span>
                    </span>
                    <span className="font-bold text-ink bg-marble px-2 py-0.5 rounded border border-hairline">
                      {totalEntities} Entities
                    </span>
                  </div>

                  {/* Skills Section */}
                  <div className="space-y-2 text-xs">
                    <div>
                      <span className="text-steel font-medium text-[11px] flex items-center gap-1 mb-1">
                        <Cpu className="w-3.5 h-3.5 text-teal-600" />
                        <span>Primary Skills ({cand.skills.length})</span>
                      </span>
                      <div className="flex flex-wrap gap-1">
                        {cand.skills.length > 0 ? (
                          cand.skills.slice(0, 5).map((s) => (
                            <span key={s} className="badge-mint px-2 py-0.5 rounded text-[11px] font-medium">
                              {s}
                            </span>
                          ))
                        ) : (
                          <span className="text-ash italic">None</span>
                        )}
                        {cand.skills.length > 5 && (
                          <span className="text-steel text-[11px] self-center">+{cand.skills.length - 5} more</span>
                        )}
                      </div>
                    </div>

                    {/* Tech Section */}
                    <div>
                      <span className="text-steel font-medium text-[11px] flex items-center gap-1 mb-1">
                        <Layers className="w-3.5 h-3.5 text-sky-600" />
                        <span>Technologies ({cand.technologies.length})</span>
                      </span>
                      <div className="flex flex-wrap gap-1">
                        {cand.technologies.length > 0 ? (
                          cand.technologies.slice(0, 5).map((t) => (
                            <span key={t} className="badge-sky px-2 py-0.5 rounded text-[11px] font-medium">
                              {t}
                            </span>
                          ))
                        ) : (
                          <span className="text-ash italic">None</span>
                        )}
                        {cand.technologies.length > 5 && (
                          <span className="text-steel text-[11px] self-center">+{cand.technologies.length - 5} more</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Actions Footer */}
                <div className="pt-4 border-t border-hairline flex items-center justify-between">
                  <button
                    onClick={() => handleDelete(cand.id, cand.name)}
                    className="text-xs text-rose-600 hover:underline font-medium flex items-center gap-1"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Delete</span>
                  </button>

                  <Link
                    href={`/candidates/${cand.id}`}
                    className="btn-primary text-xs py-1.5 px-3.5 flex items-center gap-1.5"
                  >
                    <span>View Profile</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
