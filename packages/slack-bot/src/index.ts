/**
 * Open-Inspect Slack Bot Worker
 *
 * Cloudflare Worker that handles Slack events and provides
 * a natural language interface to the coding agent.
 */

import { Hono } from "hono";
import type { Env, RepoConfig, CallbackContext, ThreadSession, UserPreferences } from "./types";
import {
  verifySlackSignature,
  postMessage,
  updateMessage,
  getChannelInfo,
  getThreadMessages,
  publishView,
} from "./utils/slack-client";
import { createClassifier } from "./classifier";
import { getAvailableRepos } from "./classifier/repos";
import { callbacksRouter } from "./callbacks";
import { generateInternalToken } from "./utils/internal";

/**
 * Build authenticated headers for control plane requests.
 */
async function getAuthHeaders(env: Env): Promise<Record<string, string>> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (env.INTERNAL_CALLBACK_SECRET) {
    const authToken = await generateInternalToken(env.INTERNAL_CALLBACK_SECRET);
    headers["Authorization"] = `Bearer ${authToken}`;
  }

  return headers;
}

/**
 * Default model when no preference is set.
 */
const DEFAULT_FALLBACK_MODEL = "claude-haiku-4-5";

/**
 * Create a session via the control plane.
 */
async function createSession(
  env: Env,
  repo: RepoConfig,
  title?: string,
  model?: string
): Promise<{ sessionId: string; status: string } | null> {
  try {
    const headers = await getAuthHeaders(env);
    const response = await env.CONTROL_PLANE.fetch("https://internal/sessions", {
      method: "POST",
      headers,
      body: JSON.stringify({
        repoOwner: repo.owner,
        repoName: repo.name,
        title: title || `Slack: ${repo.name}`,
        model: model || env.DEFAULT_MODEL || DEFAULT_FALLBACK_MODEL,
      }),
    });

    if (!response.ok) {
      console.error(`Failed to create session: ${response.status}`);
      return null;
    }

    return (await response.json()) as { sessionId: string; status: string };
  } catch (e) {
    console.error("Error creating session:", e);
    return null;
  }
}

/**
 * Send a prompt to a session via the control plane.
 */
async function sendPrompt(
  env: Env,
  sessionId: string,
  content: string,
  authorId: string,
  callbackContext?: CallbackContext
): Promise<{ messageId: string } | null> {
  try {
    const headers = await getAuthHeaders(env);
    const response = await env.CONTROL_PLANE.fetch(
      `https://internal/sessions/${sessionId}/prompt`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({
          content,
          authorId,
          source: "slack",
          callbackContext,
        }),
      }
    );

    if (!response.ok) {
      console.error(`Failed to send prompt: ${response.status}`);
      return null;
    }

    return (await response.json()) as { messageId: string };
  } catch (e) {
    console.error("Error sending prompt:", e);
    return null;
  }
}

/**
 * Generate a consistent KV key for thread-to-session mapping.
 */
function getThreadSessionKey(channel: string, threadTs: string): string {
  return `thread:${channel}:${threadTs}`;
}

/**
 * Look up an existing session for a thread.
 * Returns the session info if found and not expired.
 */
async function lookupThreadSession(
  env: Env,
  channel: string,
  threadTs: string
): Promise<ThreadSession | null> {
  try {
    const key = getThreadSessionKey(channel, threadTs);
    const data = await env.SLACK_KV.get(key, "json");
    if (data && typeof data === "object") {
      return data as ThreadSession;
    }
    return null;
  } catch (e) {
    console.error(`Error looking up thread session for channel=${channel} thread=${threadTs}:`, e);
    return null;
  }
}

/**
 * Store a session mapping for a thread.
 * TTL is 24 hours by default.
 */
async function storeThreadSession(
  env: Env,
  channel: string,
  threadTs: string,
  session: ThreadSession
): Promise<void> {
  try {
    const key = getThreadSessionKey(channel, threadTs);
    await env.SLACK_KV.put(key, JSON.stringify(session), {
      expirationTtl: 86400, // 24 hours
    });
  } catch (e) {
    console.error(`Error storing thread session for channel=${channel} thread=${threadTs}:`, e);
  }
}

/**
 * Clear a stale session mapping for a thread.
 */
async function clearThreadSession(env: Env, channel: string, threadTs: string): Promise<void> {
  try {
    const key = getThreadSessionKey(channel, threadTs);
    await env.SLACK_KV.delete(key);
  } catch (e) {
    console.error(`Error clearing thread session for channel=${channel} thread=${threadTs}:`, e);
  }
}

/**
 * Available Claude models for user selection.
 */
const AVAILABLE_MODELS = [
  { label: "Claude Haiku 4.5 (Fast)", value: "claude-haiku-4-5" },
  { label: "Claude Sonnet 4.5 (Balanced)", value: "claude-sonnet-4-5" },
  { label: "Claude Opus 4.5 (Powerful)", value: "claude-opus-4-5" },
];

/**
 * Check if a model value is valid (exists in AVAILABLE_MODELS).
 */
function isValidModel(model: string): boolean {
  return AVAILABLE_MODELS.some((m) => m.value === model);
}

/**
 * Normalize a model value to ensure it's valid.
 * Returns the model if valid, otherwise returns the fallback.
 */
function normalizeModel(model: string | undefined, fallback: string): string {
  if (model && isValidModel(model)) {
    return model;
  }
  return fallback;
}

/**
 * Generate a consistent KV key for user preferences.
 */
function getUserPreferencesKey(userId: string): string {
  return `user_prefs:${userId}`;
}

/**
 * Type guard to validate UserPreferences shape from KV.
 */
function isValidUserPreferences(data: unknown): data is UserPreferences {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return false;
  }
  const obj = data as Record<string, unknown>;
  return (
    typeof obj.userId === "string" &&
    typeof obj.model === "string" &&
    typeof obj.updatedAt === "number"
  );
}

/**
 * Look up user preferences from KV.
 */
async function getUserPreferences(env: Env, userId: string): Promise<UserPreferences | null> {
  try {
    const key = getUserPreferencesKey(userId);
    const data = await env.SLACK_KV.get(key, "json");
    if (isValidUserPreferences(data)) {
      return data;
    }
    return null;
  } catch (e) {
    console.error(`Error getting user preferences for ${userId}:`, e);
    return null;
  }
}

/**
 * Save user preferences to KV.
 * @returns true if saved successfully, false otherwise
 */
async function saveUserPreferences(env: Env, userId: string, model: string): Promise<boolean> {
  try {
    const key = getUserPreferencesKey(userId);
    const prefs: UserPreferences = {
      userId,
      model,
      updatedAt: Date.now(),
    };
    // No TTL - preferences persist indefinitely
    await env.SLACK_KV.put(key, JSON.stringify(prefs));
    return true;
  } catch (e) {
    console.error(`Error saving user preferences for ${userId}:`, e);
    return false;
  }
}

/**
 * Publish the App Home view for a user.
 */
async function publishAppHome(env: Env, userId: string): Promise<void> {
  const prefs = await getUserPreferences(env, userId);
  const fallback = env.DEFAULT_MODEL || DEFAULT_FALLBACK_MODEL;
  // Normalize model to ensure it's valid - UI and behavior will be consistent
  const currentModel = normalizeModel(prefs?.model, fallback);
  const currentModelInfo =
    AVAILABLE_MODELS.find((m) => m.value === currentModel) || AVAILABLE_MODELS[0];

  const view = {
    type: "home",
    blocks: [
      {
        type: "header",
        text: { type: "plain_text", text: "Settings" },
      },
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: "Configure your Open-Inspect preferences below.",
        },
      },
      { type: "divider" },
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: "*Model*\nSelect the Claude model for your coding sessions:",
        },
      },
      {
        type: "actions",
        block_id: "model_selection",
        elements: [
          {
            type: "static_select",
            action_id: "select_model",
            initial_option: {
              text: { type: "plain_text", text: currentModelInfo.label },
              value: currentModelInfo.value,
            },
            options: AVAILABLE_MODELS.map((m) => ({
              text: { type: "plain_text", text: m.label },
              value: m.value,
            })),
          },
        ],
      },
      { type: "divider" },
      {
        type: "context",
        elements: [
          {
            type: "mrkdwn",
            text: `Currently using: *${currentModelInfo.label}*`,
          },
        ],
      },
    ],
  };

  const result = await publishView(env.SLACK_BOT_TOKEN, userId, view);
  if (!result.ok) {
    console.error(`Failed to publish App Home for ${userId}:`, result.error);
  }
}

/**
 * Build a ThreadSession object for storage.
 */
function buildThreadSession(sessionId: string, repo: RepoConfig, model: string): ThreadSession {
  return {
    sessionId,
    repoId: repo.id,
    repoFullName: repo.fullName,
    model,
    createdAt: Date.now(),
  };
}

/**
 * Format thread context for inclusion in a prompt.
 * Returns a formatted string with previous messages from the thread.
 */
function formatThreadContext(previousMessages: string[]): string {
  if (previousMessages.length === 0) {
    return "";
  }

  const context = previousMessages.join("\n");
  return `Context from the Slack thread:\n---\n${context}\n---\n\n`;
}

/**
 * Create a session and send the initial prompt.
 * Shared logic between handleAppMention and handleRepoSelection.
 *
 * @returns Object containing sessionId if successful, null if session creation or prompt failed
 */
async function startSessionAndSendPrompt(
  env: Env,
  repo: RepoConfig,
  channel: string,
  threadTs: string,
  messageText: string,
  userId: string,
  previousMessages?: string[]
): Promise<{ sessionId: string } | null> {
  // Fetch user's preferred model and validate it
  const userPrefs = await getUserPreferences(env, userId);
  const fallback = env.DEFAULT_MODEL || DEFAULT_FALLBACK_MODEL;
  const model = normalizeModel(userPrefs?.model, fallback);

  // Create session via control plane with user's preferred model
  const session = await createSession(env, repo, messageText.slice(0, 100), model);

  if (!session) {
    await postMessage(
      env.SLACK_BOT_TOKEN,
      channel,
      "Sorry, I couldn't create a session. Please try again.",
      { thread_ts: threadTs }
    );
    return null;
  }

  await storeThreadSession(
    env,
    channel,
    threadTs,
    buildThreadSession(session.sessionId, repo, model)
  );

  // Build callback context for follow-up notification
  const callbackContext: CallbackContext = {
    channel,
    threadTs,
    repoFullName: repo.fullName,
    model,
  };

  // Build prompt content with thread context if available
  const threadContext = previousMessages ? formatThreadContext(previousMessages) : "";
  const promptContent = threadContext + messageText;

  // Send the prompt to the session
  const promptResult = await sendPrompt(
    env,
    session.sessionId,
    promptContent,
    `slack:${userId}`,
    callbackContext
  );

  if (!promptResult) {
    await postMessage(
      env.SLACK_BOT_TOKEN,
      channel,
      "Session created but failed to send prompt. Please try again.",
      { thread_ts: threadTs }
    );
    return null;
  }

  return { sessionId: session.sessionId };
}

/**
 * Post the "session started" notification to Slack.
 */
async function postSessionStartedMessage(
  env: Env,
  channel: string,
  threadTs: string,
  sessionId: string
): Promise<void> {
  await postMessage(
    env.SLACK_BOT_TOKEN,
    channel,
    `Session started! The agent is now working on your request.\n\nView progress: ${env.WEB_APP_URL}/session/${sessionId}`,
    { thread_ts: threadTs }
  );
}

const app = new Hono<{ Bindings: Env }>();

// Health check
app.get("/health", async (c) => {
  let repoCount = 0;

  try {
    const repos = await getAvailableRepos(c.env);
    repoCount = repos.length;
  } catch {
    // Control plane may be unavailable
  }

  return c.json({
    status: "healthy",
    service: "open-inspect-slack-bot",
    repoCount,
  });
});

// Slack Events API
app.post("/events", async (c) => {
  const signature = c.req.header("x-slack-signature") ?? null;
  const timestamp = c.req.header("x-slack-request-timestamp") ?? null;
  const body = await c.req.text();

  // Verify request signature
  const isValid = await verifySlackSignature(
    signature,
    timestamp,
    body,
    c.env.SLACK_SIGNING_SECRET
  );

  if (!isValid) {
    console.error("Invalid Slack signature");
    return c.json({ error: "Invalid signature" }, 401);
  }

  const payload = JSON.parse(body);

  // Handle URL verification challenge
  if (payload.type === "url_verification") {
    return c.json({ challenge: payload.challenge });
  }

  // Deduplicate events - Slack can retry on timeouts
  // Use event_id to prevent duplicate session creation
  const eventId = payload.event_id as string | undefined;
  if (eventId) {
    const dedupeKey = `event:${eventId}`;
    const existing = await c.env.SLACK_KV.get(dedupeKey);
    if (existing) {
      console.log(`Ignoring duplicate event: ${eventId}`);
      return c.json({ ok: true });
    }
    // Mark as seen with 1 hour TTL (Slack retries are within minutes)
    await c.env.SLACK_KV.put(dedupeKey, "1", { expirationTtl: 3600 });
  }

  // Process event asynchronously
  c.executionCtx.waitUntil(handleSlackEvent(payload, c.env));

  // Respond immediately (Slack requires response within 3 seconds)
  return c.json({ ok: true });
});

// Slack Interactions (buttons, modals, etc.)
app.post("/interactions", async (c) => {
  const signature = c.req.header("x-slack-signature") ?? null;
  const timestamp = c.req.header("x-slack-request-timestamp") ?? null;
  const body = await c.req.text();

  const isValid = await verifySlackSignature(
    signature,
    timestamp,
    body,
    c.env.SLACK_SIGNING_SECRET
  );

  if (!isValid) {
    return c.json({ error: "Invalid signature" }, 401);
  }

  const payloadStr = new URLSearchParams(body).get("payload") || "{}";
  const payload = JSON.parse(payloadStr);

  c.executionCtx.waitUntil(handleSlackInteraction(payload, c.env));

  return c.json({ ok: true });
});

// Mount callbacks router for control-plane notifications
app.route("/callbacks", callbacksRouter);

/**
 * Handle incoming Slack events.
 */
async function handleSlackEvent(
  payload: {
    type: string;
    event?: {
      type: string;
      text?: string;
      user?: string;
      channel?: string;
      ts?: string;
      thread_ts?: string;
      bot_id?: string;
      tab?: string;
    };
  },
  env: Env
): Promise<void> {
  if (payload.type !== "event_callback" || !payload.event) {
    return;
  }

  const event = payload.event;

  // Ignore bot messages to prevent loops
  if (event.bot_id) {
    return;
  }

  // Handle app_home_opened events
  if (event.type === "app_home_opened" && event.tab === "home" && event.user) {
    await publishAppHome(env, event.user);
    return;
  }

  // Handle app_mention events
  if (event.type === "app_mention" && event.text && event.channel && event.ts) {
    await handleAppMention(event as Required<typeof event>, env);
  }
}

/**
 * Handle app_mention events.
 */
async function handleAppMention(
  event: {
    type: string;
    text: string;
    user: string;
    channel: string;
    ts: string;
    thread_ts?: string;
  },
  env: Env
): Promise<void> {
  const { text, channel, ts, thread_ts } = event;

  // Remove the bot mention from the text
  const messageText = text.replace(/<@[A-Z0-9]+>/g, "").trim();

  if (!messageText) {
    await postMessage(
      env.SLACK_BOT_TOKEN,
      channel,
      "Hi! Please include a message with your request.",
      { thread_ts: thread_ts || ts }
    );
    return;
  }

  if (thread_ts) {
    const existingSession = await lookupThreadSession(env, channel, thread_ts);
    if (existingSession) {
      const callbackContext: CallbackContext = {
        channel,
        threadTs: thread_ts,
        repoFullName: existingSession.repoFullName,
        model: existingSession.model,
      };

      const promptResult = await sendPrompt(
        env,
        existingSession.sessionId,
        messageText,
        `slack:${event.user}`,
        callbackContext
      );

      if (promptResult) {
        return;
      }

      console.warn(
        `Failed to send to existing session ${existingSession.sessionId} for channel=${channel} thread=${thread_ts}, clearing stale mapping`
      );
      await clearThreadSession(env, channel, thread_ts);
    }
  }

  // Get channel context
  let channelName: string | undefined;
  let channelDescription: string | undefined;

  try {
    const channelInfo = await getChannelInfo(env.SLACK_BOT_TOKEN, channel);
    if (channelInfo.ok && channelInfo.channel) {
      channelName = channelInfo.channel.name;
      channelDescription = channelInfo.channel.topic?.value || channelInfo.channel.purpose?.value;
    }
  } catch {
    // Channel info not available
  }

  // Get thread context if in a thread (include bot messages for better context)
  let previousMessages: string[] | undefined;
  if (thread_ts) {
    try {
      const threadResult = await getThreadMessages(env.SLACK_BOT_TOKEN, channel, thread_ts, 10);
      if (threadResult.ok && threadResult.messages) {
        previousMessages = threadResult.messages
          .filter((m) => m.ts !== ts) // Exclude current message, but include bot messages
          .map((m) => (m.bot_id ? `[Bot]: ${m.text}` : `[User]: ${m.text}`))
          .slice(-10);
      }
    } catch {
      // Thread messages not available
    }
  }

  // Classify the repository
  const classifier = createClassifier(env);
  const result = await classifier.classify(messageText, {
    channelId: channel,
    channelName,
    channelDescription,
    threadTs: thread_ts,
    previousMessages,
  });

  // Post initial response
  if (result.needsClarification || !result.repo) {
    // Need to clarify which repo
    const repos = await getAvailableRepos(env);

    if (repos.length === 0) {
      await postMessage(
        env.SLACK_BOT_TOKEN,
        channel,
        "Sorry, no repositories are currently available. Please check that the GitHub App is installed and configured.",
        { thread_ts: thread_ts || ts }
      );
      return;
    }

    // Store original message in KV for later retrieval when user selects a repo
    const pendingKey = `pending:${channel}:${thread_ts || ts}`;
    await env.SLACK_KV.put(
      pendingKey,
      JSON.stringify({ message: messageText, userId: event.user, previousMessages }),
      { expirationTtl: 3600 } // Expire after 1 hour
    );

    // Build repo selection message
    const repoOptions = (result.alternatives || repos.slice(0, 5)).map((r) => ({
      text: {
        type: "plain_text" as const,
        text: r.displayName,
      },
      description: {
        type: "plain_text" as const,
        text: r.description.slice(0, 75),
      },
      value: r.id,
    }));

    await postMessage(
      env.SLACK_BOT_TOKEN,
      channel,
      `I couldn't determine which repository you're referring to. ${result.reasoning}`,
      {
        thread_ts: thread_ts || ts,
        blocks: [
          {
            type: "section",
            text: {
              type: "mrkdwn",
              text: `I couldn't determine which repository you're referring to.\n\n_${result.reasoning}_`,
            },
          },
          {
            type: "section",
            text: {
              type: "mrkdwn",
              text: "Which repository should I work with?",
            },
            accessory: {
              type: "static_select",
              placeholder: {
                type: "plain_text",
                text: "Select a repository",
              },
              options: repoOptions,
              action_id: "select_repo",
            },
          },
        ],
      }
    );
    return;
  }

  // We have a confident repo match - acknowledge and start session
  const { repo } = result;

  // Post initial acknowledgment
  const ackResult = await postMessage(
    env.SLACK_BOT_TOKEN,
    channel,
    `Working on *${repo.fullName}*...`,
    {
      thread_ts: thread_ts || ts,
      blocks: [
        {
          type: "section",
          text: {
            type: "mrkdwn",
            text: `Working on *${repo.fullName}*...\n_${result.reasoning}_`,
          },
        },
      ],
    }
  );

  const ackTs = ackResult.ts;
  const threadKey = thread_ts || ts;

  // Create session and send prompt using shared logic
  const sessionResult = await startSessionAndSendPrompt(
    env,
    repo,
    channel,
    threadKey,
    messageText,
    event.user,
    previousMessages
  );

  if (!sessionResult) {
    return;
  }

  // Update the acknowledgment message with session link button
  if (ackTs) {
    await updateMessage(env.SLACK_BOT_TOKEN, channel, ackTs, `Working on *${repo.fullName}*...`, {
      blocks: [
        {
          type: "section",
          text: {
            type: "mrkdwn",
            text: `Working on *${repo.fullName}*...\n_${result.reasoning}_`,
          },
        },
        {
          type: "actions",
          elements: [
            {
              type: "button",
              text: {
                type: "plain_text",
                text: "View Session",
              },
              url: `${env.WEB_APP_URL}/session/${sessionResult.sessionId}`,
              action_id: "view_session",
            },
          ],
        },
      ],
    });
  }

  // Post that the agent is working
  await postSessionStartedMessage(env, channel, threadKey, sessionResult.sessionId);
}

/**
 * Handle repo selection from clarification dropdown.
 */
async function handleRepoSelection(
  repoId: string,
  channel: string,
  messageTs: string,
  threadTs: string | undefined,
  env: Env
): Promise<void> {
  // Retrieve pending message from KV
  const pendingKey = `pending:${channel}:${threadTs || messageTs}`;
  const pendingData = await env.SLACK_KV.get(pendingKey, "json");

  if (!pendingData || typeof pendingData !== "object") {
    await postMessage(
      env.SLACK_BOT_TOKEN,
      channel,
      "Sorry, I couldn't find your original request. Please try again.",
      { thread_ts: threadTs || messageTs }
    );
    return;
  }

  const {
    message: messageText,
    userId,
    previousMessages,
  } = pendingData as {
    message: string;
    userId: string;
    previousMessages?: string[];
  };

  // Find the repo config
  const repos = await getAvailableRepos(env);
  const repo = repos.find((r) => r.id === repoId);

  if (!repo) {
    await postMessage(
      env.SLACK_BOT_TOKEN,
      channel,
      "Sorry, that repository is no longer available. Please try again.",
      { thread_ts: threadTs || messageTs }
    );
    return;
  }

  // Post acknowledgment
  await postMessage(env.SLACK_BOT_TOKEN, channel, `Working on *${repo.fullName}*...`, {
    thread_ts: threadTs || messageTs,
  });

  const threadKey = threadTs || messageTs;

  // Create session and send prompt using shared logic
  const sessionResult = await startSessionAndSendPrompt(
    env,
    repo,
    channel,
    threadKey,
    messageText,
    userId,
    previousMessages
  );

  if (!sessionResult) {
    return;
  }

  // Clean up pending message
  await env.SLACK_KV.delete(pendingKey);

  // Post that the agent is working
  await postSessionStartedMessage(env, channel, threadKey, sessionResult.sessionId);
}

/**
 * Handle Slack interactions (buttons, select menus, etc.)
 */
async function handleSlackInteraction(
  payload: {
    type: string;
    actions?: Array<{
      action_id: string;
      selected_option?: { value: string };
    }>;
    channel?: { id: string };
    message?: { ts: string; thread_ts?: string };
    user?: { id: string };
  },
  env: Env
): Promise<void> {
  if (payload.type !== "block_actions" || !payload.actions?.length) {
    return;
  }

  const action = payload.actions[0];
  const channel = payload.channel?.id;
  const messageTs = payload.message?.ts;
  const threadTs = payload.message?.thread_ts;
  const userId = payload.user?.id;

  switch (action.action_id) {
    case "select_model": {
      // Handle model selection from App Home
      const selectedModel = action.selected_option?.value;
      // Validate the selected model before saving
      if (selectedModel && userId && isValidModel(selectedModel)) {
        await saveUserPreferences(env, userId, selectedModel);
        await publishAppHome(env, userId);
      }
      break;
    }

    case "select_repo": {
      if (!channel || !messageTs) return;
      const repoId = action.selected_option?.value;
      if (repoId) {
        await handleRepoSelection(repoId, channel, messageTs, threadTs, env);
      }
      break;
    }

    case "view_session": {
      // This is a URL button, no action needed
      break;
    }
  }
}

export default app;
