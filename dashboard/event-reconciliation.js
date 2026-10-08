export function reconcileStatusElements(root, lookup, present) {
  root.querySelectorAll("[data-live-task-status]").forEach((element) => {
    const record = lookup(element.dataset.liveTaskStatus);
    if (!record) return;
    const status = present(record);
    element.textContent = status.label;
    element.className = `status-label ${status.tone}`;
  });
}

export function replaceRegionPreservingContext({ current, markup, bind, documentRoot = document, windowRoot = window }) {
  const holder = documentRoot.createElement("div");
  holder.innerHTML = markup;
  const next = holder.firstElementChild;
  if (!next || current.dataset.resultSignature === next.dataset.resultSignature) return false;
  if (current.contains(documentRoot.activeElement)) return false;
  const scrollX = windowRoot.scrollX;
  const scrollY = windowRoot.scrollY;
  current.replaceWith(next);
  bind(next);
  windowRoot.scrollTo(scrollX, scrollY);
  return true;
}

export function shouldRefreshAfterEvent({ view, activeElement, main }) {
  const editing = Boolean(activeElement && main.contains(activeElement) && activeElement.matches("input, textarea, select"));
  return view !== "workspace" && !editing;
}
