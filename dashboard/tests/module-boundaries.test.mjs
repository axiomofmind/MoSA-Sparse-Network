import test from "node:test";
import assert from "node:assert/strict";

import { shouldRefreshAfterEvent } from "../event-reconciliation.js";
import { routeFromHash } from "../shell.js";
import { renderHomeView } from "../views/home.js";
import { renderMyWorkView } from "../views/my-work.js";

test("shell routing preserves stable workspace identity", () => {
  assert.deepEqual(routeFromHash("#workspace/draft-123"), { view: "workspace", id: "draft-123" });
  assert.deepEqual(routeFromHash("#usecases"), { view: "home", id: null });
  assert.deepEqual(routeFromHash("#unknown"), { view: "home", id: null });
});

test("event reconciliation does not replace a workspace or an active editor", () => {
  const editor = { matches: () => true };
  const main = { contains: (value) => value === editor };
  assert.equal(shouldRefreshAfterEvent({ view: "workspace", activeElement: null, main }), false);
  assert.equal(shouldRefreshAfterEvent({ view: "work", activeElement: editor, main }), false);
  assert.equal(shouldRefreshAfterEvent({ view: "work", activeElement: null, main }), true);
});

test("home and My work views render independently from app orchestration", () => {
  const template = { id: "private-document-analysis", kind: "document", task_ui: {} };
  const state = { homeFiles: [], taskIntent: "", taskSuggestion: null };
  const home = renderHomeView({ state, templates: [template], drafts: [] });
  assert.match(home, /What would you like to get done/);
  const work = renderMyWorkView({
    drafts: [{ id: "draft-1", templateId: template.id, title: "Invoice", prompt: "Extract total" }],
    entries: [],
    templates: [template],
    canCancel: false,
    actionButton: () => "",
    emptyState: () => "",
  });
  assert.match(work, /data-resume-draft="draft-1"/);
});
