"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { checkApiHealth } from "@/lib/api";
import { Users, Briefcase, FileText, ArrowRight, Server, Bot } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();
  const [apiStatus, setApiStatus] = useState<"online" | "connecting" | "offline">("connecting");

  useEffect(() => {
    setApiStatus("connecting");
    checkApiHealth().then((status) => setApiStatus(status));
  }, [pathname]);

  const navItems = [
    { name: "Extract Profile", href: "/extract", icon: FileText },
    { name: "Candidates", href: "/candidates", icon: Users },
    { name: "Jobs", href: "/jobs", icon: Briefcase },
    { name: "Recruiter Assistant", href: "/assistant", icon: Bot },
  ];

  return (
    <header className="sticky top-0 z-50 w-full bg-marble border-b border-hairline">
      {/* Top Banner — Subtle & Professional */}
      <div className="bg-zinc-900 text-zinc-300 text-xs font-mono py-1.5 px-4 text-center flex items-center justify-center gap-2">
        <Server className="w-3.5 h-3.5 text-emerald-400" />
        <span>Recruitment Intelligence · Local NLP · Deterministic Matching</span>
      </div>

      {/* Primary Navigation Bar */}
      <div className="max-w-[1200px] mx-auto px-6 h-16 flex items-center justify-between">
        {/* Brand Logo — Clean Monochrome */}
        <Link href="/extract" className="flex items-center gap-2 font-semibold text-ink text-base tracking-tight hover:opacity-90">
          <div className="w-6 h-6 rounded bg-ink flex items-center justify-center text-white font-bold text-xs font-mono">
            R
          </div>
          <span className="font-semibold text-ink">Recruiter Engine</span>
        </Link>

        {/* Navigation Links */}
        <nav className="hidden md:flex items-center gap-6">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-2 text-sm font-medium transition-colors ${
                  isActive
                    ? "text-ink font-semibold border-b-2 border-ink py-4"
                    : "text-steel hover:text-ink"
                }`}
              >
                <Icon className="w-4 h-4" />
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Dynamic Backend Status Indicator */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-steel bg-zinc-50 border border-hairline px-3 py-1.5 rounded-full font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                apiStatus === "online"
                  ? "bg-emerald-500"
                  : apiStatus === "connecting"
                  ? "bg-amber-400 animate-pulse"
                  : "bg-rose-500"
              }`}
            />
            <span>
              {apiStatus === "online"
                ? "Backend Online"
                : apiStatus === "connecting"
                ? "Backend Connecting..."
                : "Backend Offline"}
            </span>
          </div>

          <Link href="/extract" className="btn-primary flex items-center gap-1.5 text-xs">
            <span>New Extraction</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>
    </header>
  );
}
