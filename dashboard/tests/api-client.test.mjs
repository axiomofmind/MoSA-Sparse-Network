import test from "node:test";
import assert from "node:assert/strict";
import { ApiError, ControllerApi } from "../api-client.js";

test("authenticated event replay sends the bearer token", async () => {
  let captured;
  globalThis.fetch = async (path, options) => {
    captured = { path, options };
    return {
      ok: true,
      text: async () => 'id: 7\nevent: ready\ndata: {"event_id":7,"event":"ready"}\n\n',
    };
  };
  const events = await new ControllerApi("viewer-token-123456").events(6);
  assert.equal(captured.path, "/v1/events?after=6&limit=1000");
  assert.equal(captured.options.headers.Authorization, "Bearer viewer-token-123456");
  assert.equal(events[0].event_id, 7);
});

test("profile application sends only plan identifiers and confirmation data", async () => {
  let captured;
  globalThis.fetch = async (path, options) => {
    captured = { path, options };
    return { ok: true, json: async () => ({ status: "completed" }) };
  };
  const api = new ControllerApi("operator-token-1234");
  const result = await api.applyProfile("plan-1", "APPLY reference-24gb", true);
  assert.equal(captured.path, "/v1/profiles/apply");
  assert.equal(captured.options.method, "POST");
  assert.deepEqual(JSON.parse(captured.options.body), {
    plan_id: "plan-1",
    confirmation: "APPLY reference-24gb",
    acknowledge_licenses: true,
  });
  assert.equal(result.status, "completed");
});

test("use-case controls use versioned controller endpoints", async () => {
  const calls = [];
  globalThis.fetch = async (path, options) => {
    calls.push({ path, options });
    return { ok: true, json: async () => ({ id: "usecase-1" }) };
  };
  const api = new ControllerApi("operator-token-1234");
  await api.createUseCase({ template_id: "developer-workstation", plan_only: true });
  await api.cancelUseCase("usecase-1", "CANCEL usecase-1");
  await api.replayUseCase("usecase-1", { confirmation: "REPLAY usecase-1" });
  assert.deepEqual(calls.map((value) => value.path), [
    "/v1/use-cases/runs",
    "/v1/use-cases/runs/usecase-1/cancel",
    "/v1/use-cases/runs/usecase-1/replay",
  ]);
  assert.equal(JSON.parse(calls[1].options.body).confirmation, "CANCEL usecase-1");
});

test("readiness is checked before a use-case run", async () => {
  const calls = [];
  globalThis.fetch = async (path, options) => {
    calls.push({ path, options });
    return { ok: true, json: async () => ({ state: "ready" }) };
  };
  const api = new ControllerApi("operator-token");
  const payload = { template_id: "private-document-analysis", inputs: [] };
  await api.useCaseReadiness(payload);
  assert.equal(calls[0].path, "/v1/use-cases/readiness");
  assert.equal(calls[0].options.method, "POST");
  assert.deepEqual(JSON.parse(calls[0].options.body), payload);
});

test("review and export operations are version-bound", async () => {
  const calls = [];
  globalThis.fetch = async (path, options) => {
    calls.push({ path, options });
    return { ok: true, json: async () => ({}) };
  };
  const api = new ControllerApi("operator-token");
  await api.correctUseCase("usecase-one", {
    version_id: "usecase-one:v1",
    changes: [{ field_id: "$text", value: "Reviewed" }],
  });
  await api.exportUseCase("usecase-one", {
    version_id: "usecase-one:v2",
    format: "report",
  });
  assert.equal(calls[0].path, "/v1/use-cases/runs/usecase-one/corrections");
  assert.equal(calls[1].path, "/v1/use-cases/runs/usecase-one/exports");
  assert.equal(JSON.parse(calls[1].options.body).version_id, "usecase-one:v2");
});

test("API errors preserve structured controller details for local presentation", async () => {
  globalThis.fetch = async () => ({
    ok: false,
    status: 400,
    statusText: "Bad Request",
    json: async () => ({
      schema: "sparse-network-error.v1",
      error: "source path is outside the approved roots",
      type: "ValueError",
    }),
  });
  await assert.rejects(
    () => new ControllerApi("operator-token").useCaseReadiness({}),
    (error) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.type, "ValueError");
      assert.equal(error.payload.schema, "sparse-network-error.v1");
      return true;
    },
  );
});
