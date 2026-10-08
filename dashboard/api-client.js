/** @typedef {{event_id:number,event:string,endpoint:string,details:Record<string,unknown>}} ControllerEvent */

export class ApiError extends Error {
  constructor(message, status = 0, payload = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
    this.type = payload?.type || null;
  }
}

export class ControllerApi {
  constructor(token = "") {
    this.token = token;
  }

  setToken(token) {
    this.token = token;
  }

  async json(path, { authenticated = true, method = "GET", body } = {}) {
    const headers = { Accept: "application/json" };
    if (authenticated && this.token) headers.Authorization = `Bearer ${this.token}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const response = await fetch(path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: "no-store",
    });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      let errorPayload = {};
      try {
        errorPayload = await response.json();
        detail = errorPayload.error || detail;
      } catch { /* use status text */ }
      throw new ApiError(detail, response.status, errorPayload);
    }
    return response.json();
  }

  health() { return this.json("/v1/health", { authenticated: false }); }
  session() { return this.json("/v1/session"); }
  snapshot() { return this.json("/v1/dashboard"); }
  artifact(id) { return this.json(`/v1/artifacts/${encodeURIComponent(id)}`); }
  artifactContent(id) { return this.json(`/v1/artifacts/${encodeURIComponent(id)}/content`); }
  run(id) { return this.json(`/v1/runs/${encodeURIComponent(id)}`); }
  cancelRequest(id, confirmation) { return this.json(`/v1/requests/${encodeURIComponent(id)}/cancel`, { method: "POST", body: { confirmation } }); }
  fleetOperation(operation, endpoint, confirmation = "") {
    return this.json(`/v1/fleet/operations/${encodeURIComponent(operation)}`, {
      method: "POST", body: { endpoint, confirmation },
    });
  }
  planProfile(profileId, customProfile = null) {
    return this.json("/v1/profiles/plan", {
      method: "POST",
      body: customProfile ? { custom_profile: customProfile } : { profile_id: profileId },
    });
  }
  applyProfile(planId, confirmation, acknowledgeLicenses) {
    return this.json("/v1/profiles/apply", {
      method: "POST", body: { plan_id: planId, confirmation, acknowledge_licenses: acknowledgeLicenses },
    });
  }
  updateConfig(changes, confirmation) {
    return this.json("/v1/config", { method: "PATCH", body: { changes, confirmation } });
  }
  createEvaluation(suite, baselines) {
    return this.json("/v1/evaluations", { method: "POST", body: { suite, baselines } });
  }
  createUseCase(payload) {
    return this.json("/v1/use-cases/runs", { method: "POST", body: payload });
  }
  useCaseReadiness(payload) {
    return this.json("/v1/use-cases/readiness", { method: "POST", body: payload });
  }
  correctUseCase(id, payload) {
    return this.json(`/v1/use-cases/runs/${encodeURIComponent(id)}/corrections`, {
      method: "POST", body: payload,
    });
  }
  exportUseCase(id, payload) {
    return this.json(`/v1/use-cases/runs/${encodeURIComponent(id)}/exports`, {
      method: "POST", body: payload,
    });
  }
  useCase(id) { return this.json(`/v1/use-cases/runs/${encodeURIComponent(id)}`); }
  cancelUseCase(id, confirmation) {
    return this.json(`/v1/use-cases/runs/${encodeURIComponent(id)}/cancel`, { method: "POST", body: { confirmation } });
  }
  replayUseCase(id, payload) {
    return this.json(`/v1/use-cases/runs/${encodeURIComponent(id)}/replay`, { method: "POST", body: payload });
  }
  decideUseCaseAdmission(id, payload) {
    return this.json(`/v1/use-cases/runs/${encodeURIComponent(id)}/admission`, { method: "POST", body: payload });
  }

  /** @returns {Promise<ControllerEvent[]>} */
  async events(after = 0) {
    const response = await fetch(`/v1/events?after=${after}&limit=1000`, {
      headers: {
        Accept: "text/event-stream",
        Authorization: `Bearer ${this.token}`,
      },
      cache: "no-store",
    });
    if (!response.ok) throw new ApiError(`Event stream returned ${response.status}`, response.status);
    const text = await response.text();
    return text.split("\n\n").flatMap((block) => {
      const data = block.split("\n").find((line) => line.startsWith("data: "));
      if (!data) return [];
      try { return [JSON.parse(data.slice(6))]; } catch { return []; }
    });
  }
}
