import test from "node:test";
import assert from "node:assert/strict";
import { clearDrafts, deleteDraft, listDrafts, loadDraft, saveDraft } from "../draft-store.js";

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

test("persists serializable draft data and marks attachments for reselection", () => {
  const storage = memoryStorage();
  const saved = saveDraft(storage, {
    id: "draft-one", templateId: "private-document-analysis", prompt: "Extract totals",
    files: [{ name: "invoice.pdf", size: 42, type: "application/pdf", lastModified: 7 }],
  }, { now: 1_000 });
  assert.equal(saved.files[0].needsReselection, true);
  assert.equal(loadDraft(storage, "draft-one", { now: 2_000 }).prompt, "Extract totals");
});

test("drops expired or corrupt drafts and supports deletion", () => {
  const storage = memoryStorage();
  saveDraft(storage, { id: "draft-one", files: [] }, { now: 0, retentionDays: 1 });
  assert.deepEqual(listDrafts(storage, { now: 86_400_001 }), []);
  saveDraft(storage, { id: "draft-two", files: [] }, { now: 100_000_000 });
  deleteDraft(storage, "draft-two", { now: 100_000_001 });
  assert.equal(loadDraft(storage, "draft-two", { now: 100_000_001 }), null);
  clearDrafts(storage);
});
