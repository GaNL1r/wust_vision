function showTab(tab) {
  document.getElementById("video-tab").style.display = tab === "video" ? "flex" : "none";
}

document.addEventListener("DOMContentLoaded", () => {
  updateCharts();
  updateMainRange();
  setInterval(() => {
    fetchDataAndUpdateCharts();
    fetchAndDisplayJsonWithTree("json-target", "/target_log");
    fetchAndDisplayJsonWithTree("json-serial", "/serial_log");
  }, 200);
});
