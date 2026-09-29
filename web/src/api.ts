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
