window.addEventListener("message", (event) => {
  if (event.data?.type !== "liesel:visualization-height") return;
  const frame = document.querySelector(
    `[data-visualization="${event.data.visualization}"]`
  );
  if (
    !frame ||
    event.source !== frame.contentWindow?.frames[0] ||
    !Number.isFinite(event.data.height) ||
    event.data.height < 0 ||
    event.data.height > 10000
  ) return;
  frame.style.height = `${Math.ceil(event.data.height) + 32}px`;
});
