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
      </body>
    </html>
  );
}
