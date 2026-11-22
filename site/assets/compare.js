// Before/after sliders: <div class="compare"> with two images; drag, click, or use the arrow keys.
document.querySelectorAll(".compare").forEach((box) => {
  const range = box.querySelector("input[type=range]");
  const set = (v) => box.style.setProperty("--pos", v + "%");
  set(range.value);
  range.addEventListener("input", () => set(range.value));
});
