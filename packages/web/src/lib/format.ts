/**
 * Utility functions for formatting display values
 */

/**
 * Format model ID to display name
 * e.g., "claude-sonnet-4-5" → "Claude Sonnet 4.5"
 * e.g., "claude-haiku-4-5" → "Claude Haiku 4.5"
 * e.g., "claude-opus-4-5" → "Claude Opus 4.5"
 */
export function formatModelName(modelId: string): string {
  if (!modelId) return "Unknown Model";

  // Handle common Claude model patterns
  const match = modelId.match(/^claude-(\w+)-(\d+)-(\d+)$/i);
  if (match) {
    const [, variant, major, minor] = match;
    const capitalizedVariant = variant.charAt(0).toUpperCase() + variant.slice(1).toLowerCase();
    return `Claude ${capitalizedVariant} ${major}.${minor}`;
  }

  // Fallback: capitalize words and replace hyphens with spaces
  return modelId
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Format model ID to lowercase display format for footer
 * e.g., "claude-sonnet-4-5" → "claude sonnet 4.5"
 */
export function formatModelNameLower(modelId: string): string {
  if (!modelId) return "unknown model";

  const match = modelId.match(/^claude-(\w+)-(\d+)-(\d+)$/i);
  if (match) {
    const [, variant, major, minor] = match;
    return `claude ${variant.toLowerCase()} ${major}.${minor}`;
  }

  return modelId.replace(/-/g, " ").toLowerCase();
}

/**
 * Truncate branch name with ellipsis at start
 * e.g., "feature/very-long-branch-name-here" → "...long-branch-name-here"
 */
export function truncateBranch(branchName: string, maxLength = 30): string {
  if (!branchName) return "";
  if (branchName.length <= maxLength) return branchName;
  return "..." + branchName.slice(-maxLength);
}

/**
 * Copy text to clipboard
 * Returns true if successful, false otherwise
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
    // Fallback for older browsers
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-999999px";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    const success = document.execCommand("copy");
    textArea.remove();
    return success;
  } catch {
    return false;
  }
}

/**
 * Format file path for display (show basename or last N characters)
 */
export function formatFilePath(
  filePath: string,
  maxLength = 40
): { display: string; full: string } {
  if (!filePath) return { display: "", full: "" };

  const parts = filePath.split("/");
  const basename = parts[parts.length - 1];

  if (basename.length <= maxLength) {
    return { display: basename, full: filePath };
  }

  return {
    display: basename.slice(0, maxLength - 3) + "...",
    full: filePath,
  };
}

/**
 * Format number with +/- prefix for diff stats
 */
export function formatDiffStat(
  additions: number,
  deletions: number
): { additions: string; deletions: string } {
  return {
    additions: additions > 0 ? `+${additions}` : "+0",
    deletions: deletions > 0 ? `-${deletions}` : "-0",
  };
}
