"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { extractEntities, saveCandidate, uploadResumeFile, ExtractedEntities } from "@/lib/api";
import {
  FileText,
  Upload,
  Code,
  CheckCircle2,
  UserPlus,
  Cpu,
  Layers,
  Globe,
  AlertCircle,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Info,
  Sliders,
  File,
  X
} from "lucide-react";

export default function ExtractPage() {
  const router = useRouter();

  const sampleSentence1 = "I worked in the AI/ML Department and worked with CNN Models using Python";
  const sampleSentence2 = "Senior Full Stack Developer with 4 years of experience building microservices in Node.js, React, Next.js, and PostgreSQL using Docker and AWS.";

  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [extracted, setExtracted] = useState<ExtractedEntities | null>(null);

  // File upload state — separate from extraction state
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [isStale, setIsStale] = useState(false); // true when file changed after extraction

  // Toggle States
  const [showJson, setShowJson] = useState(false);
  const [showPipelineDetails, setShowPipelineDetails] = useState(false);

  // Save Candidate Modal State
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [candidateName, setCandidateName] = useState("");
  const [contactInfo, setContactInfo] = useState("");
  const [saving, setSaving] = useState(false);

  const handleExtract = async () => {
    if (!text.trim()) {
      setError("Please enter a candidate description or upload a resume file.");
      return;
    }
    setError(null);
    setIsStale(false);
    setLoading(true);
    try {
      const res = await extractEntities(text);
      setExtracted(res);
    } catch (err: any) {
      setError(err.message || "Failed to extract entities");
    } finally {
      setLoading(false);
    }
  };

  /**
   * File upload handler — ONLY extracts text from PDF/TXT.
   * Does NOT call the NLP extraction endpoint.
   * Clears any previously extracted result to prevent showing stale data.
   */
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Clear stale extraction immediately
    if (extracted) {
      setExtracted(null);
      setIsStale(false);
    }

    setError(null);
    setUploading(true);
    setUploadedFileName(null);

    try {
      const res = await uploadResumeFile(file);
      setText(res.extracted_text);
      setUploadedFileName(res.filename);
    } catch (err: any) {
      setError(err.message || "File upload failed");
    } finally {
      setUploading(false);
    }

    // Reset the input so the same file can be re-selected if needed
    e.target.value = "";
  };

  /**
   * When text is manually edited after extraction, mark result as potentially stale.
   */
  const handleTextChange = (val: string) => {
    setText(val);
    if (extracted) {
      setIsStale(true);
    }
  };

  const handleClearFile = () => {
    setUploadedFileName(null);
    setText("");
    setExtracted(null);
    setIsStale(false);
    setError(null);
  };

  const handleSaveProfile = async () => {
    if (!candidateName.trim()) {
      alert("Please enter candidate name.");
      return;
    }
    if (!extracted) return;

    setSaving(true);
    try {
      const res = await saveCandidate({
        name: candidateName,
        contact_info: contactInfo,
        raw_text: text,
        skills: extracted.skill,
        technologies: extracted.technology,
        languages: extracted.language,
      });
      setShowSaveModal(false);
      if (res.is_duplicate) {
        alert(res.message || "Candidate profile already exists. Redirecting to existing profile.");
      }
      router.push(`/candidates/${res.id}`);
    } catch (err: any) {
      alert(err.message || "Failed to save profile.");
    } finally {
      setSaving(false);
    }
  };

  const totalEntities = extracted
    ? extracted.skill.length + extracted.technology.length + extracted.language.length
    : 0;

  return (
    <div className="space-y-8">
      {/* Section Header */}
      <div className="border-b border-hairline pb-6">
        <div className="flex items-center gap-2 text-xs font-mono text-steel uppercase tracking-wider mb-1">
          Part 1 — NLP Entity Extraction
        </div>
        <h1 className="text-3xl font-semibold tracking-tight text-ink">
          Candidate Profile Extraction
        </h1>
        <p className="text-steel text-sm mt-1 max-w-3xl">
          Convert unstructured candidate descriptions into normalized, canonical skill profiles using local gazetteer and phrase segmentation.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column — Dual Input Layout */}
        <div className="lg:col-span-7 space-y-6">
          <div className="card-grafbase p-6 space-y-6">

            {/* Input Option A: Conversational Description */}
            <div className="space-y-3">
              <div>
                <h3 className="text-sm font-semibold text-ink">Option A: Describe the Candidate</h3>
                <p className="text-xs text-steel">Tell us about their background, skills, technologies, and projects in plain text.</p>
              </div>

              {/* Stale result warning */}
              {isStale && extracted && (
                <div className="p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-xs flex items-center gap-2">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>Text changed after extraction. Click <strong>Extract Profile &amp; Entities</strong> to refresh results.</span>
                </div>
              )}

              <textarea
                rows={7}
                className="w-full p-4 rounded-xl border border-hairline bg-canvas/30 text-ink placeholder:text-ash text-xs focus:outline-none focus:ring-2 focus:ring-ink focus:border-transparent font-mono leading-relaxed"
                placeholder="e.g. 'I worked in the AI/ML Department and worked with CNN Models using Python...'"
                value={text}
                onChange={(e) => handleTextChange(e.target.value)}
              />
            </div>

            {/* Input Option B: Resume Upload */}
            <div className="pt-4 border-t border-hairline space-y-3">
              <div>
                <h4 className="text-xs font-semibold text-ink">Option B: Upload Resume Document</h4>
                <p className="text-[11px] text-steel">PDF or TXT format · Max 5 MB · Text will be loaded into the field above for review before extraction.</p>
              </div>

              {uploadedFileName ? (
                /* File loaded state — show filename + clear button */
                <div className="flex items-center gap-3 p-3 rounded-xl border border-hairline bg-canvas/40">
                  <File className="w-4 h-4 text-steel shrink-0" />
                  <span className="text-xs text-ink font-mono flex-1 truncate">{uploadedFileName}</span>
                  <span className="text-[10px] text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full font-medium">
                    Loaded
                  </span>
                  <button
                    type="button"
                    onClick={handleClearFile}
                    className="text-ash hover:text-rose-600 transition-colors p-0.5"
                    title="Clear file"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ) : (
                /* Upload button */
                <label className={`cursor-pointer btn-ghost text-xs py-2 px-4 flex items-center gap-2 w-fit ${uploading ? "opacity-60 pointer-events-none" : ""}`}>
                  <Upload className="w-4 h-4 text-steel" />
                  <span>{uploading ? "Reading file..." : "Choose File"}</span>
                  <input
                    type="file"
                    accept=".pdf,.txt"
                    className="hidden"
                    onChange={handleFileUpload}
                    disabled={uploading}
                  />
                </label>
              )}
            </div>

            {/* Try an Example Presets */}
            <div className="flex items-center gap-2 pt-1 border-t border-hairline flex-wrap">
              <span className="text-xs text-steel font-medium">Try an example:</span>
              <button
                type="button"
                onClick={() => {
                  setText(sampleSentence1);
                  setUploadedFileName(null);
                  if (extracted) setIsStale(true);
                }}
                className="text-xs btn-pill text-steel hover:text-ink"
              >
                [AI/ML + CNN + Python]
              </button>
              <button
                type="button"
                onClick={() => {
                  setText(sampleSentence2);
                  setUploadedFileName(null);
                  if (extracted) setIsStale(true);
                }}
                className="text-xs btn-pill text-steel hover:text-ink"
              >
                [Full Stack Web]
              </button>
            </div>

            {error && (
              <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center justify-between pt-2 gap-3 flex-wrap">
              <button
                type="button"
                onClick={handleExtract}
                disabled={loading || !text.trim()}
                className="btn-primary flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span>{loading ? "Running NLP Pipeline..." : "Extract Profile & Entities"}</span>
              </button>

              {extracted && !isStale && (
                <button
                  type="button"
                  onClick={() => setShowSaveModal(true)}
                  className="btn-ghost flex items-center gap-2 text-xs"
                >
                  <UserPlus className="w-4 h-4 text-emerald-600" />
                  <span>Save Candidate Profile</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right Column — Extracted Profile & Pipeline Disclosure */}
        <div className="lg:col-span-5 space-y-6">
          <div className="card-grafbase p-6 space-y-6">
            <div className="flex items-center justify-between border-b border-hairline pb-4">
              <div>
                <span className="text-[11px] font-mono text-steel uppercase">Extraction Output</span>
                <h3 className="font-semibold text-ink text-base">Extracted Profile</h3>
              </div>

              {extracted && !isStale && (
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold text-ink bg-canvas px-2.5 py-1 rounded-full border border-hairline">
                    {totalEntities} entities extracted
                  </span>
                  <button
                    type="button"
                    onClick={() => setShowJson(!showJson)}
                    className="text-xs btn-pill flex items-center gap-1"
                  >
                    <Code className="w-3.5 h-3.5" />
                    <span>{showJson ? "Cards" : "JSON"}</span>
                  </button>
                </div>
              )}
            </div>

            {loading ? (
              <div className="py-12 text-center text-steel space-y-3">
                <div className="w-6 h-6 border-2 border-steel border-t-transparent rounded-full animate-spin mx-auto" />
                <p className="text-sm font-mono">Running NLP pipeline...</p>
                <p className="text-xs text-ash">spaCy · Gazetteer · Synonym normalization</p>
              </div>
            ) : isStale && extracted ? (
              <div className="py-12 text-center text-steel space-y-3">
                <AlertCircle className="w-8 h-8 mx-auto text-amber-400 stroke-[1.5]" />
                <p className="text-sm">Input changed.</p>
                <p className="text-xs text-ash max-w-xs mx-auto">
                  Click <strong>Extract Profile & Entities</strong> to update results for the new text.
                </p>
              </div>
            ) : !extracted ? (
              <div className="py-12 text-center text-steel space-y-3">
                <Info className="w-8 h-8 mx-auto text-ash stroke-[1.5]" />
                <p className="text-sm">No profile extracted yet.</p>
                <p className="text-xs text-ash max-w-xs mx-auto">
                  {uploadedFileName
                    ? `"${uploadedFileName}" loaded. Click Extract Profile & Entities to analyse it.`
                    : "Type a description or upload a resume, then click Extract Profile & Entities."}
                </p>
              </div>
            ) : showJson ? (
              <pre className="bg-zinc-900 text-emerald-400 p-4 rounded-xl text-xs font-mono overflow-x-auto max-h-[380px] leading-relaxed">
                {JSON.stringify(extracted, null, 2)}
              </pre>
            ) : (
              <div className="space-y-5">
                {/* Skills Category */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-steel font-medium border-b border-hairline pb-1">
                    <span className="flex items-center gap-1.5 font-semibold text-ink">
                      <Cpu className="w-3.5 h-3.5 text-teal-600" />
                      <span>Skills</span>
                    </span>
                    <span className="font-mono text-ink font-semibold">{extracted.skill.length}</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 min-h-[32px]">
                    {extracted.skill.length > 0 ? (
                      extracted.skill.map((sk) => (
                        <span key={sk} className="badge-mint px-2.5 py-1 rounded text-xs font-medium">
                          {sk}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-ash italic">No skills detected</span>
                    )}
                  </div>
                </div>

                {/* Technologies Category */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-steel font-medium border-b border-hairline pb-1">
                    <span className="flex items-center gap-1.5 font-semibold text-ink">
                      <Layers className="w-3.5 h-3.5 text-sky-600" />
                      <span>Technologies &amp; Tools</span>
                    </span>
                    <span className="font-mono text-ink font-semibold">{extracted.technology.length}</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 min-h-[32px]">
                    {extracted.technology.length > 0 ? (
                      extracted.technology.map((tech) => (
                        <span key={tech} className="badge-sky px-2.5 py-1 rounded text-xs font-medium">
                          {tech}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-ash italic">No technologies detected</span>
                    )}
                  </div>
                </div>

                {/* Languages Category */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-steel font-medium border-b border-hairline pb-1">
                    <span className="flex items-center gap-1.5 font-semibold text-ink">
                      <Globe className="w-3.5 h-3.5 text-lime-600" />
                      <span>Languages</span>
                    </span>
                    <span className="font-mono text-ink font-semibold">{extracted.language.length}</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 min-h-[32px]">
                    {extracted.language.length > 0 ? (
                      extracted.language.map((lang) => (
                        <span key={lang} className="badge-moss px-2.5 py-1 rounded text-xs font-medium">
                          {lang}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-ash italic">No languages detected</span>
                    )}
                  </div>
                </div>

                {/* NLP Pipeline Transparency Panel */}
                <div className="pt-4 border-t border-hairline">
                  <button
                    type="button"
                    onClick={() => setShowPipelineDetails(!showPipelineDetails)}
                    className="w-full flex items-center justify-between text-xs text-steel hover:text-ink font-medium"
                  >
                    <span className="flex items-center gap-1.5">
                      <Sliders className="w-3.5 h-3.5 text-ink" />
                      <span>How it was extracted (NLP Pipeline Details)</span>
                    </span>
                    {showPipelineDetails ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                  </button>

                  {showPipelineDetails && extracted.pipeline_details && (
                    <div className="mt-3 p-3.5 rounded-xl bg-canvas/60 border border-hairline space-y-3 text-xs font-mono">
                      <div>
                        <span className="text-ash uppercase text-[10px]">Segmentation Method</span>
                        <p className="text-ink font-sans text-xs">{extracted.pipeline_details.method}</p>
                      </div>

                      {extracted.pipeline_details.synonym_mappings.length > 0 && (
                        <div>
                          <span className="text-ash uppercase text-[10px]">Synonym Canonical Mappings</span>
                          <div className="space-y-1 mt-1">
                            {extracted.pipeline_details.synonym_mappings.map((m, idx) => (
                              <div key={idx} className="flex justify-between items-center text-[11px] bg-marble p-1.5 rounded border border-hairline">
                                <span className="text-steel">"{m.raw_phrase}"</span>
                                <span className="text-ink font-bold">→ {m.canonical}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {extracted.pipeline_details.noun_chunks.length > 0 && (
                        <div>
                          <span className="text-ash uppercase text-[10px]">Extracted Noun Chunks</span>
                          <p className="text-steel text-[11px] truncate">
                            {extracted.pipeline_details.noun_chunks.join(", ")}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Save Footer */}
                <div className="pt-2 border-t border-hairline flex items-center justify-between">
                  <span className="text-xs text-steel font-mono">Canonical Normalization Verified</span>
                  <button
                    type="button"
                    onClick={() => setShowSaveModal(true)}
                    className="btn-primary text-xs py-2 px-4 flex items-center gap-1.5"
                  >
                    <span>Save Profile & Continue</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Save Candidate Profile Modal */}
      {showSaveModal && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="card-grafbase max-w-md w-full p-6 space-y-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-hairline pb-3">
              <h3 className="font-semibold text-ink text-base">Save Candidate Profile</h3>
              <button onClick={() => setShowSaveModal(false)} className="text-steel hover:text-ink text-sm">
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-steel mb-1">
                  Candidate Name *
                </label>
                <input
                  type="text"
                  placeholder="e.g. Priya Sharma"
                  value={candidateName}
                  onChange={(e) => setCandidateName(e.target.value)}
                  className="w-full p-2.5 rounded-lg border border-hairline bg-canvas/30 text-sm focus:outline-none focus:ring-2 focus:ring-ink"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-steel mb-1">
                  Contact Info / Email (Optional)
                </label>
                <input
                  type="text"
                  placeholder="e.g. priya@example.com"
                  value={contactInfo}
                  onChange={(e) => setContactInfo(e.target.value)}
                  className="w-full p-2.5 rounded-lg border border-hairline bg-canvas/30 text-sm focus:outline-none focus:ring-2 focus:ring-ink"
                />
              </div>

              <div className="p-3 rounded-lg bg-zinc-50 border border-hairline text-xs space-y-1 font-mono">
                <span className="font-semibold text-ink font-sans">Extracted Profile Summary:</span>
                <p className="text-steel">
                  {extracted?.skill.length || 0} skills · {extracted?.technology.length || 0} tech tools · {extracted?.language.length || 0} languages
                </p>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 pt-3 border-t border-hairline">
              <button
                type="button"
                onClick={() => setShowSaveModal(false)}
                className="btn-ghost text-xs py-2 px-4"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSaveProfile}
                disabled={saving}
                className="btn-primary text-xs py-2 px-4"
              >
                {saving ? "Saving..." : "Save Candidate"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
