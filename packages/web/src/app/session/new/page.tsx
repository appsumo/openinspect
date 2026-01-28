"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { SidebarLayout, useSidebarContext } from "@/components/sidebar-layout";

interface Repo {
  id: number;
  fullName: string;
  owner: string;
  name: string;
  description: string | null;
  private: boolean;
}

export default function NewSessionPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [repos, setRepos] = useState<Repo[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [selectedRepo, setSelectedRepo] = useState<string>("");
  const [title, setTitle] = useState("");
  const [selectedModel, setSelectedModel] = useState<string>("claude-haiku-4-5");
  const [error, setError] = useState("");

  const models = [
    { id: "claude-haiku-4-5", name: "Claude Haiku 4.5", description: "Fast & affordable" },
    { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", description: "Balanced performance" },
  ];

  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/");
    }
  }, [status, router]);

  useEffect(() => {
    if (session) {
      fetchRepos();
    }
  }, [session]);

  const fetchRepos = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/repos");
      if (res.ok) {
        const data = await res.json();
        setRepos(data.repos || []);
      }
    } catch (error) {
      console.error("Failed to fetch repos:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedRepo) {
      setError("Please select a repository");
      return;
    }

    setCreating(true);
    setError("");

    const [owner, name] = selectedRepo.split("/");

    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repoOwner: owner,
          repoName: name,
          title: title || undefined,
          model: selectedModel,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        router.push(`/session/${data.sessionId}`);
      } else {
        const data = await res.json();
        setError(data.error || "Failed to create session");
      }
    } catch (_error) {
      setError("Failed to create session");
    } finally {
      setCreating(false);
    }
  };

  if (status === "loading" || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-foreground" />
      </div>
    );
  }

  return (
    <SidebarLayout>
      <NewSessionContent
        repos={repos}
        selectedRepo={selectedRepo}
        setSelectedRepo={setSelectedRepo}
        title={title}
        setTitle={setTitle}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
        models={models}
        error={error}
        creating={creating}
        handleSubmit={handleSubmit}
      />
    </SidebarLayout>
  );
}

function NewSessionContent({
  repos,
  selectedRepo,
  setSelectedRepo,
  title,
  setTitle,
  selectedModel,
  setSelectedModel,
  models,
  error,
  creating,
  handleSubmit,
}: {
  repos: Repo[];
  selectedRepo: string;
  setSelectedRepo: (value: string) => void;
  title: string;
  setTitle: (value: string) => void;
  selectedModel: string;
  setSelectedModel: (value: string) => void;
  models: { id: string; name: string; description: string }[];
  error: string;
  creating: boolean;
  handleSubmit: (e: React.FormEvent) => void;
}) {
  const { isOpen, toggle } = useSidebarContext();

  return (
    <div className="h-full flex flex-col">
      {/* Header with toggle when sidebar is closed */}
      {!isOpen && (
        <header className="border-b border-border-muted flex-shrink-0">
          <div className="px-4 py-3 flex items-center gap-3">
            <button
              onClick={toggle}
              className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-muted transition"
              title="Open sidebar"
            >
              <SidebarToggleIcon />
            </button>
            <h1 className="text-lg font-semibold text-foreground">New Session</h1>
          </div>
        </header>
      )}

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-4 py-8">
          {isOpen && <h1 className="text-2xl font-bold text-foreground mb-8">New Session</h1>}
          <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
              <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 px-4 py-3 border border-red-200 dark:border-red-800">
                {error}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">Repository</label>
              <select
                value={selectedRepo}
                onChange={(e) => setSelectedRepo(e.target.value)}
                className="w-full px-4 py-3 border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                required
              >
                <option value="">Select a repository...</option>
                {repos.map((repo) => (
                  <option key={repo.id} value={repo.fullName} className="text-foreground bg-input">
                    {repo.fullName} {repo.private ? "(private)" : ""}
                  </option>
                ))}
              </select>
              {repos.length === 0 && (
                <p className="mt-2 text-sm text-muted-foreground">
                  No repositories found. Make sure you have granted access to your repositories.
                </p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Title (optional)
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g., Add user authentication"
                className="w-full px-4 py-3 border border-border bg-transparent text-foreground focus:outline-none focus:ring-2 focus:ring-ring placeholder:text-secondary-foreground"
              />
              <p className="mt-2 text-sm text-muted-foreground">
                A title helps identify the session. If not provided, the repository name will be
                used.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-4 py-3 border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              >
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} - {model.description}
                  </option>
                ))}
              </select>
              <p className="mt-2 text-sm text-muted-foreground">
                Haiku is faster and more affordable. Sonnet provides better reasoning for complex
                tasks.
              </p>
            </div>

            <button
              type="submit"
              disabled={creating || !selectedRepo}
              className="w-full py-3 bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {creating ? "Creating..." : "Create Session"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

function SidebarToggleIcon() {
  return (
    <svg
      className="w-4 h-4"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <line x1="9" y1="3" x2="9" y2="21" />
    </svg>
  );
}
