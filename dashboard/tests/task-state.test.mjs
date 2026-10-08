import test from "node:test";
import assert from "node:assert/strict";
import { taskState, taskStateFromEvent } from "../task-state.js";

test("maps controller execution states without conflating checks and review", () => {
  const value = taskState({
    state: "completed",
    result: {
      state: "needs_human_review",
      verification: { accepted: true, checks: [{ name: "citations", passed: true }] },
    },
  });
  assert.equal(value.label, "Needs review");
  assert.equal(value.execution, "completed");
  assert.equal(value.verification, "passed");
  assert.equal(value.review, "required");
});

test("maps terminal and unknown states honestly", () => {
  assert.equal(taskState({ state: "failed" }).label, "Could not finish");
  assert.equal(taskState({ state: "cancelled" }).label, "Cancelled");
  assert.equal(taskState({ state: "surprising" }).label, "Status unavailable");
  assert.equal(taskState({ state: "surprising" }).known, false);
});

test("only maps events that establish a user-facing transition", () => {
  assert.equal(taskStateFromEvent({ event: "use_case_preparing_models" }), "preparing_models");
  assert.equal(taskStateFromEvent({ event: "telemetry_sample" }), null);
});
