import test from "node:test";
import assert from "node:assert/strict";
import {
  fleetEndpointStatus,
  formatBytes,
  graphColumns,
  groupSwapEvents,
  percent,
  profileForSelection,
} from "../model.js";

test("formats resource values and clamps percentages", () => {
  assert.equal(formatBytes(1024 ** 3), "1.0 GiB");
  assert.equal(percent(15, 10), 100);
  assert.equal(percent(1, 0), 0);
});

test("selects automatic and explicit portable profiles", () => {
  const catalog = {
    recommended_selection: "24",
    profiles: [{ id: "six", selection_key: "24" }, { id: "small", selection_key: "16" }],
    custom: { id: "custom-preview", selection_key: "custom" },
  };
  assert.equal(profileForSelection(catalog, "auto").id, "six");
  assert.equal(profileForSelection(catalog, "16").id, "small");
  assert.equal(profileForSelection(catalog, "custom").id, "custom-preview");
});

test("prefers the active roster when a hardware tier has variants", () => {
  const catalog = {
    profiles: [
      { id: "32-gemma", selection_key: "32", active: false },
      { id: "32-reference", selection_key: "32", active: true },
    ],
  };
  assert.equal(profileForSelection(catalog, "32").id, "32-reference");
  assert.equal(profileForSelection(catalog, "32-gemma").id, "32-gemma");
});

test("auto-selects the standard variant when the recommended tier is inactive", () => {
  const catalog = {
    recommended_selection: "32",
    profiles: [
      { id: "32-lean", selection_key: "32", variant: "lean", active: false },
      { id: "32-standard", selection_key: "32", variant: "standard", active: false },
    ],
  };
  assert.equal(profileForSelection(catalog, "auto").id, "32-standard");
});

test("labels installed optional endpoints as available on demand", () => {
  const status = fleetEndpointStatus(
    "pp-ocrv6-medium",
    { state: "unavailable" },
    { artifact_state: "validated" },
    { optional: ["pp-ocrv6-medium"] },
  );
  assert.deepEqual(status, {
    label: "available on demand",
    tone: "neutral",
    onDemand: true,
  });
  assert.equal(
    fleetEndpointStatus(
      "missing-ocr",
      { state: "unavailable" },
      { artifact_state: "missing_or_invalid" },
      { optional: ["missing-ocr"] },
    ).label,
    "unavailable",
  );
});

test("groups ordered swap phases without inventing transitions", () => {
  const grouped = groupSwapEvents([
    { event_id: 2, event: "swap_phase", execution_id: "s1", details: { phase: "healthy" } },
    { event_id: 1, event: "swap_phase", execution_id: "s1", details: { phase: "draining" } },
    { event_id: 3, event: "request_completed", execution_id: "x", details: {} },
  ]);
  assert.equal(grouped.length, 1);
  assert.deepEqual(grouped[0].events.map((event) => event.details.phase), ["draining", "healthy"]);
});

test("lays a bounded execution graph into dependency columns", () => {
  const columns = graphColumns({ nodes: [
    { id: "route", depends_on: [] },
    { id: "invoke", depends_on: ["route"] },
    { id: "verify", depends_on: ["invoke"] },
  ] });
  assert.deepEqual(columns.map((column) => column.map((node) => node.id)), [["route"], ["invoke"], ["verify"]]);
});
