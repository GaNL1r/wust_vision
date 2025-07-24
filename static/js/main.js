function showTab(tab) {
  document.getElementById("video-tab").style.display = tab === "video" ? "flex" : "none";
}

document.addEventListener("DOMContentLoaded", () => {
  updateCharts();
  updateMainRange();
  setInterval(() => {
    fetchDataAndUpdateCharts();
    fetchAndDisplayJsonWithTree("json-serial", "/serial_log");
    fetchAndDisplayJsonWithTree("json-target", "/target_log");
  }, 200);
});
