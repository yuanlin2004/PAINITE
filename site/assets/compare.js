// Before/after sliders (<div class="compare">): drag anywhere on the photo, or
// tab to it and use the arrow keys. The hidden range input keeps it accessible.
document.querySelectorAll(".compare").forEach((box) => {
  const range = box.querySelector("input[type=range]");
  const set = (v) => {
    v = Math.max(0, Math.min(100, v));
    range.value = v;
    box.style.setProperty("--pos", v + "%");
  };
  const fromPointer = (e) => {
    const r = box.getBoundingClientRect();
    set(((e.clientX - r.left) / r.width) * 100);
  };
  let dragging = false;
  box.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    dragging = true;
    box.setPointerCapture(e.pointerId);
    fromPointer(e);
    e.preventDefault();
  });
  box.addEventListener("pointermove", (e) => { if (dragging) fromPointer(e); });
  const stop = (e) => {
    dragging = false;
    if (box.hasPointerCapture(e.pointerId)) box.releasePointerCapture(e.pointerId);
  };
  box.addEventListener("pointerup", stop);
  box.addEventListener("pointercancel", stop);
  range.addEventListener("input", () => set(Number(range.value)));
  set(Number(range.value));
});
