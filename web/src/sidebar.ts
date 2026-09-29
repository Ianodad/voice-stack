import type { Conversation } from "./api";

export interface SidebarHandlers {
  onSelect(id: string): void;
  onDelete(id: string): void;
}

function when(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { month: "short", day: "numeric" });
}

export class Sidebar {
  private items: Conversation[] = [];
  private activeId: string | null = null;

  constructor(
    private list: HTMLElement,
    private empty: HTMLElement,
    private handlers: SidebarHandlers,
  ) {}

  setConversations(items: Conversation[]): void {
    this.items = items;
    this.render();
  }

  setActive(id: string | null): void {
    this.activeId = id;
    this.render();
  }

  newest(): Conversation | undefined {
    return this.items[0];
  }

  private render(): void {
    this.list.replaceChildren();
    this.empty.hidden = this.items.length > 0;
    for (const c of this.items) {
      const li = document.createElement("li");
      li.className = "conv" + (c.id === this.activeId ? " active" : "");
      const open = document.createElement("button");
      open.type = "button";
      open.className = "conv-open";
      if (c.id === this.activeId) open.setAttribute("aria-current", "true");
      const title = document.createElement("span");
      title.className = "conv-title";
      title.textContent = c.title || "New conversation";
      const time = document.createElement("span");
      time.className = "conv-time";
      time.textContent = when(c.updated_at);
      open.append(title, time);
      open.addEventListener("click", () => this.handlers.onSelect(c.id));
      const del = document.createElement("button");
      del.type = "button";
      del.className = "conv-del";
      del.textContent = "×";
      del.setAttribute("aria-label", `Delete conversation: ${c.title || "New conversation"}`);
      del.addEventListener("click", () => this.handlers.onDelete(c.id));
      li.append(open, del);
      this.list.append(li);
    }
  }
}
