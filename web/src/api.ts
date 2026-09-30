export interface Conversation {
  id: string;
  title: string;
  updated_at: string;
}
export interface StoredMessage {
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

const JSON_HEADERS = { "Content-Type": "application/json" };

async function ok(res: Response): Promise<Response> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res;
}

export async function listConversations(): Promise<Conversation[]> {
  return (await ok(await fetch("/api/conversations"))).json();
}

export async function getConversation(id: string): Promise<StoredMessage[]> {
  return (await ok(await fetch(`/api/conversations/${encodeURIComponent(id)}`))).json();
}

export async function deleteConversation(id: string): Promise<void> {
  await ok(await fetch(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }));
}

export async function restartLlm(): Promise<void> {
  await ok(await fetch("/api/llm/restart", { method: "POST", headers: JSON_HEADERS, body: "{}" }));
}

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  constructor(status: number, detail: string) {
    super(`${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

async function actionPost(id: string, verb: "approve" | "deny"): Promise<void> {
  const res = await fetch(`/api/actions/${encodeURIComponent(id)}/${verb}`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: "{}",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const d = (await res.json()).detail;
      if (typeof d === "string") detail = d;
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, detail);
  }
}

export const approveAction = (id: string) => actionPost(id, "approve");
export const denyAction = (id: string) => actionPost(id, "deny");

export async function pendingActions(): Promise<unknown[]> {
  const res = await fetch("/api/actions/pending");
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  const data = await res.json();
  return Array.isArray(data) ? data : [];
}
