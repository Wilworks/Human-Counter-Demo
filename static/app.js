let currentMode = "frontal_depth";

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("total-count").innerText = data.total_count;
    document.getElementById("active-count").innerText = data.active_tracks;
    document.getElementById("count-in").innerText = data.count_in;
    document.getElementById("count-out").innerText = data.count_out;
    document.getElementById("fps-counter").innerText = data.fps.toFixed(1);
    document.getElementById("res-display").innerText = `${data.width}x${data.height}`;
    document.getElementById("gate-mode-label").innerText = data.mode.toUpperCase();

    // Update active button state
    currentMode = data.mode;
    document.getElementById("btn-mode-frontal").className = currentMode === "frontal_depth" ? "btn active" : "btn";
    document.getElementById("btn-mode-sideways").className = currentMode === "sideways_wall" ? "btn active" : "btn";

    // Update Telemetry Table
    const tbody = document.getElementById("telemetry-table-body");
    const telemetry = data.telemetry || {};
    const tids = Object.keys(telemetry);

    if (tids.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="color: var(--text-muted); text-align: center;">NO ACTIVE TRACKS</td></tr>';
      return;
    }

    let html = "";
    for (const tid of tids) {
      const item = telemetry[tid];
      const isCounted = item.is_counted;
      const statusBadge = isCounted 
        ? '<span class="badge-passed">PASSED</span>' 
        : '<span class="badge-tracking">TRACKING</span>';

      html += `
        <tr>
          <td>#${item.entity_id}</td>
          <td>#${tid}</td>
          <td>${item.depth_m}m</td>
          <td>${item.angle_deg}°</td>
          <td>${statusBadge}</td>
        </tr>
      `;
    }
    tbody.innerHTML = html;

  } catch (err) {
    console.error("Error fetching status:", err);
  }
}

async function setMode(mode) {
  try {
    await fetch("/api/update_config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode })
    });
    fetchStatus();
  } catch (err) {
    console.error("Error updating mode:", err);
  }
}

async function updateConfig() {
  const depth = parseFloat(document.getElementById("slider-depth").value);
  const reid = parseFloat(document.getElementById("slider-reid").value);

  document.getElementById("val-depth").innerText = depth.toFixed(1) + "m";
  document.getElementById("val-reid").innerText = reid.toFixed(2);

  try {
    await fetch("/api/update_config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_depth_meters: depth,
        reid_similarity_threshold: reid
      })
    });
  } catch (err) {
    console.error("Error updating config:", err);
  }
}

async function resetMemory() {
  try {
    await fetch("/api/reset", { method: "POST" });
    fetchStatus();
  } catch (err) {
    console.error("Error resetting memory:", err);
  }
}

function exportLog() {
  window.open("/output/human_crossing_log.csv", "_blank");
}

// Poll telemetry every 250ms
setInterval(fetchStatus, 250);
fetchStatus();
