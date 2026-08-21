import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "AI Recruiter — NLP Extraction & Deterministic Matching",
  description: "AI-assisted recruitment tool converting conversational text to structured profiles and deterministic fit scoring.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-canvas text-ink antialiased flex flex-col min-h-screen">
        <Navbar />
        <main className="flex-1 max-w-[1200px] w-full mx-auto p-6 md:p-8">
          {children}
        </main>
        <footer className="w-full border-t border-hairline bg-marble py-6 px-6 mt-12 text-center text-xs text-steel">
          <div className="max-w-[1200px] mx-auto flex flex-col sm:flex-row justify-between items-center gap-4">
            <div>
              <span className="font-semibold text-ink">AI Recruiter</span> — Local NLP & Matching Platform
            </div>
            <div className="flex gap-4 text-steel font-mono">
              <span>Part 1: Extraction</span>
              <span>•</span>
              <span>Part 2: Deterministic Scoring</span>
              <span>•</span>
              <span>Zero LLM API Cost</span>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
