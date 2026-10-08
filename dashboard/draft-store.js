export const DRAFT_SCHEMA = "sparse-dashboard-drafts.v1";
export const DEFAULT_RETENTION_DAYS = 30;
const STORAGE_KEY = "sparse-dashboard-drafts";

function nowIso(now = Date.now()) {
  return new Date(now).toISOString();
}

export function createDraftId(randomUUID = globalThis.crypto?.randomUUID?.bind(globalThis.crypto)) {
  if (randomUUID) return `draft-${randomUUID()}`;
  return `draft-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function serializableDraft(draft) {
  const copy = { ...draft };
  copy.files = (draft.files || []).map((file) => ({
    name: String(file.name || "attachment"),
    size: Number(file.size || 0),
    type: String(file.type || "application/octet-stream"),
    lastModified: Number(file.lastModified || 0),
    needsReselection: true,
  }));
  return copy;
}

function readEnvelope(storage, now = Date.now()) {
  try {
    const parsed = JSON.parse(storage.getItem(STORAGE_KEY) || "null");
    if (!parsed || parsed.schema !== DRAFT_SCHEMA || !Array.isArray(parsed.drafts)) return [];
    return parsed.drafts.filter((entry) => {
      const expires = Date.parse(entry.expiresAt || "");
      return Number.isFinite(expires) && expires > now && entry.draft?.id;
    });
  } catch {
    return [];
  }
}

function writeEnvelope(storage, drafts) {
  storage.setItem(STORAGE_KEY, JSON.stringify({ schema: DRAFT_SCHEMA, drafts }));
}

export function saveDraft(storage, draft, { now = Date.now(), retentionDays = DEFAULT_RETENTION_DAYS } = {}) {
  const id = draft.id || createDraftId();
  const updatedAt = nowIso(now);
  const expiresAt = nowIso(now + retentionDays * 86_400_000);
  const value = serializableDraft({ ...draft, id, updatedAt });
  const entries = readEnvelope(storage, now).filter((entry) => entry.draft.id !== id);
  entries.push({ updatedAt, expiresAt, draft: value });
  entries.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  writeEnvelope(storage, entries);
  return value;
}

export function loadDraft(storage, id, { now = Date.now() } = {}) {
  return readEnvelope(storage, now).find((entry) => entry.draft.id === id)?.draft || null;
}

export function listDrafts(storage, { now = Date.now() } = {}) {
  const entries = readEnvelope(storage, now);
  writeEnvelope(storage, entries);
  return entries.map((entry) => entry.draft);
}

export function deleteDraft(storage, id, { now = Date.now() } = {}) {
  const entries = readEnvelope(storage, now).filter((entry) => entry.draft.id !== id);
  writeEnvelope(storage, entries);
}

export function clearDrafts(storage) {
  storage.removeItem(STORAGE_KEY);
}
