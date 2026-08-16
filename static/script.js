// CityEye — Frontend Dashboard Script

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const liveClockEl = document.getElementById("liveClock");
  const alertBanner = document.getElementById("alertBanner");
  const alertTitle = document.getElementById("alertTitle");
  const alertMessage = document.getElementById("alertMessage");
  const alertCloseBtn = document.getElementById("alertCloseBtn");

  // Stats Elements
  const statVehicles = document.getElementById("statVehicles");
  const statPersons = document.getElementById("statPersons");
  const statTriple = document.getElementById("statTriple");
  const statWrongWay = document.getElementById("statWrongWay");
  const statStopped = document.getElementById("statStopped");
  const statHelmet = document.getElementById("statHelmet");
  const helmetMeta = document.getElementById("helmetMeta");
  const eventTotalBadge = document.getElementById("eventTotalBadge");

  // Action Buttons
  const btnRunProcess = document.getElementById("btnRunProcess");
  const btnRefresh = document.getElementById("btnRefresh");
  const btnSampleGen = document.getElementById("btnSampleGen");

  // Video and Table
  const videoPlayer = document.getElementById("processedVideoPlayer");
  const videoPlaceholder = document.getElementById("videoPlaceholder");
  const eventsTableBody = document.getElementById("eventsTableBody");
  const eventFilterSelect = document.getElementById("eventFilterSelect");

  let allEvents = [];

  // Update Clock
  function updateClock() {
    const now = new Date();
    liveClockEl.textContent = now.toTimeString().split(" ")[0] + " " + (now.toLocaleDateString());
  }
  setInterval(updateClock, 1000);
  updateClock();

  // Show Alert
  function showAlert(title, message, type = "warning") {
    alertBanner.className = `alert-banner ${type}`;
    alertTitle.textContent = title;
    alertMessage.textContent = message;
    alertBanner.classList.remove("hidden");
  }

  alertCloseBtn.addEventListener("click", () => {
    alertBanner.classList.add("hidden");
  });

  // Fetch Health & System State
  async function fetchHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();

      if (data.helmet_detection_enabled) {
        helmetMeta.textContent = "AI Model: Active";
      } else {
        helmetMeta.textContent = "Modular (Model not configured)";
      }

      if (data.processed_output_available) {
        videoPlaceholder.classList.add("hidden");
        videoPlayer.style.display = "block";
      } else {
        videoPlaceholder.classList.remove("hidden");
      }
    } catch (err) {
      console.error("Health check error:", err);
    }
  }

  // Fetch Events & Update Stats
  async function fetchEvents() {
    try {
      const res = await fetch("/events");
      const data = await res.json();

      allEvents = data.events || [];
      const stats = data.statistics || {};

      // Update Metric Badges
      statVehicles.textContent = stats.total_vehicles || 0;
      statPersons.textContent = stats.total_persons || 0;
      statTriple.textContent = stats.triple_riding || 0;
      statWrongWay.textContent = stats.wrong_way_driving || 0;
      statStopped.textContent = stats.vehicle_stopped || 0;
      statHelmet.textContent = stats.helmet_violation || 0;
      eventTotalBadge.textContent = `${data.total_events || 0} Events`;

      renderEventsTable();
    } catch (err) {
      console.error("Error fetching events:", err);
    }
  }

  // Render Events Table
  function renderEventsTable() {
    const filter = eventFilterSelect.value;
    const filtered = filter === "ALL" 
      ? allEvents 
      : allEvents.filter(e => e.event === filter);

    eventsTableBody.innerHTML = "";

    if (filtered.length === 0) {
      eventsTableBody.innerHTML = `
        <tr class="empty-row">
          <td colspan="6">No events matching "${filter}"</td>
        </tr>
      `;
      return;
    }

    filtered.forEach(ev => {
      const tr = document.createElement("tr");

      // Format time
      let timeStr = ev.timestamp || "N/A";
      if (timeStr.includes("T")) {
        timeStr = timeStr.split("T")[1];
      }

      // Format details
      let details = [];
      if (ev.person_count) details.push(`Riders: ${ev.person_count}`);
      if (ev.movement_direction) details.push(`Heading: ${ev.movement_direction}`);
      if (ev.stopped_duration_sec) details.push(`Stopped: ${ev.stopped_duration_sec}s`);
      if (ev.details && typeof ev.details === 'object') {
        for (const [k, v] of Object.entries(ev.details)) {
          details.push(`${k}: ${v}`);
        }
      }
      const detailsStr = details.join(" | ") || "Traffic Violation";

      const pillClass = `pill-${ev.event || 'default'}`;
      const eventLabel = (ev.event || "UNKNOWN").replace(/_/g, " ");

      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-size: 0.75rem;">${timeStr}</td>
        <td><code>${ev.camera_id || 'cam_01'}</code></td>
        <td><span class="event-pill ${pillClass}">${eventLabel}</span></td>
        <td><strong>#${ev.vehicle_id !== undefined ? ev.vehicle_id : 'N/A'}</strong></td>
        <td><span style="color: var(--neon-cyan);">${ev.confidence ? (ev.confidence * 100).toFixed(0) + '%' : '90%'}</span></td>
        <td style="color: var(--text-muted); font-size: 0.78rem;">${detailsStr}</td>
      `;
      eventsTableBody.appendChild(tr);
    });
  }

  eventFilterSelect.addEventListener("change", renderEventsTable);

  // Trigger Video Process Action
  btnRunProcess.addEventListener("click", async () => {
    btnRunProcess.disabled = true;
    btnRunProcess.innerHTML = `
      <svg class="spinner" viewBox="0 0 50 50" style="width:16px;height:16px;animation:spin 1s linear infinite;">
        <circle cx="25" cy="25" r="20" fill="none" stroke="currentColor" stroke-width="5" stroke-dasharray="31.4 31.4"></circle>
      </svg>
      Processing Video...
    `;

    try {
      const res = await fetch("/process", { method: "POST" });
      const data = await res.json();

      if (!res.ok || data.status === "error") {
        showAlert("Video Processing Error", data.message || "Failed to process video.", "error");
      } else {
        showAlert("Success", `Video analysis complete! Detected ${data.events_count} event(s).`, "success");
        // Reload video player with fresh timestamp to bypass cache
        videoPlaceholder.classList.add("hidden");
        videoPlayer.style.display = "block";
        videoPlayer.src = `/output-video?t=${Date.now()}`;
        videoPlayer.load();
        videoPlayer.play().catch(() => {});
      }

      await fetchEvents();
      await fetchHealth();
    } catch (err) {
      showAlert("Execution Error", err.message || "Could not connect to backend server.", "error");
    } finally {
      btnRunProcess.disabled = false;
      btnRunProcess.innerHTML = `
        <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        Run AI Video Analysis
      `;
    }
  });

  // Generate Sample Video Action
  btnSampleGen.addEventListener("click", async () => {
    btnSampleGen.disabled = true;
    btnSampleGen.textContent = "Generating...";

    try {
      const res = await fetch("/create-sample-video", { method: "POST" });
      const data = await res.json();

      if (res.ok && data.status === "success") {
        showAlert("Sample Video Ready", "Synthetic CCTV test video generated in videos/traffic.mp4! Now click 'Run AI Video Analysis'.", "success");
      } else {
        showAlert("Sample Generation Failed", data.message || "Error generating video.", "error");
      }
    } catch (err) {
      showAlert("Error", err.message, "error");
    } finally {
      btnSampleGen.disabled = false;
      btnSampleGen.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
          <line x1="7" y1="2" x2="7" y2="22"></line>
          <line x1="17" y1="2" x2="17" y2="22"></line>
          <line x1="2" y1="12" x2="22" y2="12"></line>
          <line x1="2" y1="7" x2="7" y2="7"></line>
          <line x1="2" y1="17" x2="7" y2="17"></line>
          <line x1="17" y1="17" x2="22" y2="17"></line>
          <line x1="17" y1="7" x2="22" y2="7"></line>
        </svg>
        Generate Sample Video
      `;
    }
  });

  // Refresh Button
  btnRefresh.addEventListener("click", () => {
    fetchEvents();
    fetchHealth();
  });

  // Initial Load
  fetchHealth();
  fetchEvents();

  // Polling every 6s
  setInterval(fetchEvents, 6000);
});
