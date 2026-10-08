import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";

const [baseUrl, token, browserPath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"] = process.argv.slice(2);
if (!baseUrl || !token) {
  throw new Error("usage: node scripts/smoke_dashboard_browser.mjs <base-url> <token> [browser-path]");
}

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function freePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => server.listen(0, "127.0.0.1", resolve).once("error", reject));
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function waitFor(callback, { timeout = 10_000, interval = 100 } = {}) {
  const deadline = Date.now() + timeout;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const value = await callback();
      if (value) return value;
    } catch (error) {
      lastError = error;
    }
    await delay(interval);
  }
  throw lastError || new Error("browser check timed out");
}

class Cdp {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.sequence = 0;
    this.pending = new Map();
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.method === "Runtime.exceptionThrown") {
        const detail = message.params?.exceptionDetails;
        process.stderr.write(`Browser exception: ${detail?.text || "unknown"} ${detail?.exception?.description || ""}\n`);
      }
      if (!message.id) return;
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error.message));
      else pending.resolve(message.result);
    });
  }

  call(method, params = {}) {
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const result = await this.call("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }

  close() { this.socket.close(); }
}

const debugPort = await freePort();
const profilePath = await mkdtemp(path.join(os.tmpdir(), "sparse-browser-smoke-"));
const browser = spawn(browserPath, [
  "--headless=new",
  "--disable-gpu",
  "--no-first-run",
  "--no-default-browser-check",
  "--remote-allow-origins=*",
  `--remote-debugging-port=${debugPort}`,
  `--user-data-dir=${profilePath}`,
  "about:blank",
], { stdio: "ignore" });

let cdp;
try {
  const pages = await waitFor(async () => {
    const response = await fetch(`http://127.0.0.1:${debugPort}/json/list`);
    const values = await response.json();
    return values.find((value) => value.type === "page" && value.webSocketDebuggerUrl);
  });
  cdp = new Cdp(pages.webSocketDebuggerUrl);
  await cdp.open();
  await cdp.call("Runtime.enable");
  await cdp.call("Page.enable");
  await cdp.call("Page.navigate", { url: `${baseUrl}/dashboard/` });
  await waitFor(() => cdp.evaluate(`location.origin === ${JSON.stringify(baseUrl)} && document.readyState === 'complete'`));
  await cdp.evaluate(`sessionStorage.setItem("sparse-dashboard-token", ${JSON.stringify(token)}); location.reload(); true`);
  await waitFor(() => cdp.evaluate("Boolean(document.querySelector('#home-task-form'))"));

  await cdp.call("Emulation.setDeviceMetricsOverride", {
    width: 360,
    height: 800,
    deviceScaleFactor: 1,
    mobile: true,
  });
  for (let index = 0; index < 24; index += 1) {
    await cdp.call("Input.dispatchKeyEvent", { type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
    await cdp.call("Input.dispatchKeyEvent", { type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
    if (await cdp.evaluate("document.activeElement?.id === 'home-intent'")) break;
  }
  const baseline = await cdp.evaluate(`(() => {
    const field = document.querySelector('#home-intent');
    const support = document.querySelector('.starter-card small');
    const surface = document.querySelector('.starter-card');
    const rgb = (value) => (value.match(/[\\d.]+/g) || []).slice(0, 3).map(Number);
    const luminance = (value) => {
      const channels = rgb(value).map((channel) => {
        const normalized = channel / 255;
        return normalized <= .04045 ? normalized / 12.92 : ((normalized + .055) / 1.055) ** 2.4;
      });
      return .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2];
    };
    const foreground = luminance(getComputedStyle(support).color);
    const background = luminance(getComputedStyle(surface).backgroundColor);
    const contrast = (Math.max(foreground, background) + .05) / (Math.min(foreground, background) + .05);
    return {
      noHorizontalOverflow: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
      bodyFont: Number.parseFloat(getComputedStyle(document.body).fontSize),
      supportFont: Number.parseFloat(getComputedStyle(document.querySelector('.starter-card small')).fontSize),
      visibleFocus: document.activeElement === field && getComputedStyle(field).outlineStyle !== 'none',
      activeId: document.activeElement?.id || document.activeElement?.textContent?.trim().slice(0, 30),
      outlineStyle: getComputedStyle(field).outlineStyle,
      supportContrast: contrast,
      navigation: [...document.querySelectorAll('#primary-nav .nav-item')].map((value) => value.textContent.trim()),
    };
  })()`);
  await cdp.call("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-motion", value: "reduce" }],
  });
  const reducedMotion = await cdp.evaluate(`(() => {
    const probe = document.createElement('div');
    probe.className = 'spinner';
    document.body.append(probe);
    const duration = Number.parseFloat(getComputedStyle(probe).animationDuration) || 0;
    probe.remove();
    return duration <= .01;
  })()`);

  await cdp.evaluate(`(() => {
    const field = document.querySelector('#home-intent');
    field.value = 'draft survives polling';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    field.focus();
    field.setSelectionRange(5, 5);
    return fetch('/v1/evaluations', {
      method: 'POST',
      headers: { Authorization: ${JSON.stringify(`Bearer ${token}`)}, 'Content-Type': 'application/json' },
      body: JSON.stringify({ suite: 'browser-focus-home', baselines: [] }),
    }).then((response) => response.ok);
  })()`);
  await delay(2_500);
  const homeRetention = await cdp.evaluate(`(() => {
    const field = document.querySelector('#home-intent');
    return {
      focused: document.activeElement === field,
      value: field.value,
      selection: field.selectionStart,
    };
  })()`);

  await cdp.evaluate(`(() => {
    const field = document.querySelector('#home-intent');
    field.value = 'Find why this Python test fails';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    document.querySelector('#home-task-form').requestSubmit();
    return true;
  })()`);
  await waitFor(
    () => cdp.evaluate("location.hash.startsWith('#workspace/') && Boolean(document.querySelector('#task-followup-form'))"),
    { timeout: 15_000 },
  );
  const firstTurn = await cdp.evaluate(`(() => ({
    turns: document.querySelectorAll('.conversation-turn').length,
    request: document.querySelector('.user-turn .turn-content')?.textContent || '',
    result: document.querySelector('.sparse-turn .result-text')?.textContent || '',
  }))()`);
  await cdp.evaluate(`(() => {
    const field = document.querySelector('#task-followup');
    field.value = 'Narrow that down to one function';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    field.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    return true;
  })()`);
  await waitFor(
    () => cdp.evaluate("document.querySelectorAll('.conversation-turn').length === 2 && Boolean(document.querySelector('#task-followup-form'))"),
    { timeout: 15_000 },
  );
  const continuation = await cdp.evaluate(`(() => ({
    turns: document.querySelectorAll('.conversation-turn').length,
    prompts: [...document.querySelectorAll('.user-turn .turn-content')].map((value) => value.textContent.trim()),
    context: document.querySelector('#task-followup-form label small')?.textContent || '',
  }))()`);
  await cdp.evaluate(`document.querySelector('[data-change-task]').click(); true`);
  await waitFor(() => cdp.evaluate("location.hash === '#home' && Boolean(document.querySelector('#home-task-form'))"));

  await cdp.evaluate(`document.querySelector('[data-home-template="private-document-analysis"]').click(); true`);
  await waitFor(() => cdp.evaluate("location.hash.startsWith('#workspace/') && Boolean(document.querySelector('#task-workspace-form'))"));
  await cdp.evaluate(`(() => {
    const field = document.querySelector('[data-task-field="prompt"]');
    document.querySelector('.edit-request').open = true;
    field.value = 'workspace survives polling';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    field.focus();
    field.setSelectionRange(9, 9);
    document.querySelector('.source-additional').open = true;
    return fetch('/v1/evaluations', {
      method: 'POST',
      headers: { Authorization: ${JSON.stringify(`Bearer ${token}`)}, 'Content-Type': 'application/json' },
      body: JSON.stringify({ suite: 'browser-focus-workspace', baselines: [] }),
    }).then((response) => response.ok);
  })()`);
  await delay(2_500);
  const workspaceRetention = await cdp.evaluate(`(() => {
    const field = document.querySelector('[data-task-field="prompt"]');
    return {
      focused: document.activeElement === field,
      value: field.value,
      selection: field.selectionStart,
      disclosureOpen: document.querySelector('.source-additional').open,
      noHorizontalOverflow: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
    };
  })()`);

  await cdp.call("Emulation.setPageScaleFactor", { pageScaleFactor: 2 });
  const zoomReadable = await cdp.evaluate("document.querySelector('#task-workspace-form').getBoundingClientRect().width > 0");
  const checks = {
    workOrientedNavigation: JSON.stringify(baseline.navigation) === JSON.stringify(["Home", "My work", "Settings"]),
    readableType: baseline.bodyFont >= 16 && baseline.supportFont >= 13,
    visibleFocus: baseline.visibleFocus,
    supportContrast: baseline.supportContrast >= 4.5,
    reducedMotion,
    mobileNoOverflow: baseline.noHorizontalOverflow && workspaceRetention.noHorizontalOverflow,
    homeFocusRetention: homeRetention.focused && homeRetention.value === "draft survives polling" && homeRetention.selection === 5,
    workspaceFocusRetention: workspaceRetention.focused && workspaceRetention.value === "workspace survives polling" && workspaceRetention.selection === 9,
    disclosureRetention: workspaceRetention.disclosureOpen,
    zoomReadable,
    automaticTaskStart: firstTurn.turns === 1
      && firstTurn.request.includes("Find why this Python test fails")
      && firstTurn.result.includes("Find why this Python test fails"),
    contextualFollowUp: continuation.turns === 2
      && continuation.prompts[1] === "Narrow that down to one function"
      && continuation.context.includes("previous result"),
  };
  const report = {
    schema: "sparse-network-dashboard-browser-smoke.v1",
    passed: Object.values(checks).every(Boolean),
    checks,
    diagnostics: {
      activeId: baseline.activeId,
      outlineStyle: baseline.outlineStyle,
      homeRetention,
      firstTurn,
      continuation,
      workspaceRetention,
    },
  };
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!report.passed) process.exitCode = 1;
} finally {
  cdp?.close();
  if (!browser.killed) browser.kill();
  await Promise.race([
    new Promise((resolve) => browser.once("exit", resolve)),
    delay(2_000),
  ]);
  try { await rm(profilePath, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 }); }
  catch { /* Windows may retain a short-lived browser lock after process exit. */ }
}
