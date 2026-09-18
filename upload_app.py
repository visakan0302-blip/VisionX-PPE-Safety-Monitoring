"""
=============================================================================
VISIONX - BROWSER-BASED VIDEO ANALYSIS INTERFACE (upload_app.py)
Local Hackathon Demo - FastAPI & Uvicorn

Features:
1. Web interface: "VISIONX — WORKER SAFETY MONITOR"
2. Upload construction video file (MP4, AVI, MOV, MKV)
3. Shared detection/risk pipeline using VisionXProcessor (models/best.pt)
4. Outputs annotated video, summary statistics, risk scoring, reasons
5. Automatic evidence generation (violation_snapshots/ and violation_logs/safety_events.csv)
6. Clean, responsive dark professional dashboard
=============================================================================
"""

import os
import shutil
import time
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

# Reuse the unified core processor and safety risk engine
from video_processor import VisionXProcessor, DEVICE, MODEL_PATH, CONFIDENCE

# =============================================================================
# INITIALIZE FASTAPI APP & STORAGE
# =============================================================================

app = FastAPI(title="VisionX - Worker Safety Monitor", version="4.1")

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
SNAPSHOTS_DIR = "violation_snapshots"
LOGS_DIR = "violation_logs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize the reusable processor once for the web server
print("=" * 60)
print("VisionX - Browser Video Analysis Application")
print(f"Loading detection & risk engine on device: {DEVICE}")
processor = VisionXProcessor(model_path=MODEL_PATH, device=DEVICE, confidence=CONFIDENCE)
print("VisionX engine ready.")
print("=" * 60)


# =============================================================================
# FRONTEND HTML TEMPLATE
# =============================================================================

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VisionX — Worker Safety Monitor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0d14;
      --card-bg: #131924;
      --card-border: #232c3d;
      --accent-blue: #388bfd;
      --accent-cyan: #00d2ff;
      --accent-green: #2ea043;
      --accent-amber: #f0883e;
      --accent-red: #f85149;
      --text-main: #f0f6fc;
      --text-muted: #8b949e;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    body {
      background-color: var(--bg-dark);
      color: var(--text-main);
      line-height: 1.5;
      padding: 24px 16px;
      min-height: 100vh;
    }

    .container {
      max-width: 1180px;
      margin: 0 auto;
    }

    header {
      text-align: center;
      margin-bottom: 28px;
    }

    .brand-title {
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #ffffff 30%, var(--accent-cyan) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 6px;
    }

    .brand-subtitle {
      color: var(--text-muted);
      font-size: 1.05rem;
      margin-bottom: 12px;
    }

    .badge-bar {
      display: flex;
      gap: 10px;
      justify-content: center;
      flex-wrap: wrap;
    }

    .badge {
      background: #1c2333;
      border: 1px solid #2e384d;
      color: #94a3b8;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
    }

    .badge.live {
      border-color: rgba(46, 160, 67, 0.4);
      color: #3fb950;
    }

    .upload-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 28px;
      text-align: center;
      margin-bottom: 28px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
      transition: border-color 0.2s;
    }

    .drop-zone {
      border: 2px dashed #2f3b52;
      border-radius: 8px;
      padding: 32px 16px;
      cursor: pointer;
      background: rgba(15, 22, 36, 0.5);
      transition: all 0.2s ease;
    }

    .drop-zone:hover, .drop-zone.dragover {
      border-color: var(--accent-blue);
      background: rgba(56, 139, 253, 0.05);
    }

    .drop-icon {
      font-size: 2.2rem;
      margin-bottom: 8px;
    }

    .drop-text {
      font-weight: 600;
      font-size: 1rem;
      margin-bottom: 4px;
    }

    .drop-subtext {
      color: var(--text-muted);
      font-size: 0.82rem;
    }

    #fileInput {
      display: none;
    }

    .file-pill {
      display: none;
      margin-top: 14px;
      background: #1e293b;
      border: 1px solid #334155;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.85rem;
      color: #cbd5e1;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    .action-btn {
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: #ffffff;
      border: none;
      padding: 12px 28px;
      font-size: 0.95rem;
      font-weight: 700;
      border-radius: 8px;
      cursor: pointer;
      margin-top: 20px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
      transition: all 0.2s;
    }

    .action-btn:hover {
      background: linear-gradient(135deg, #3b82f6, #2563eb);
      box-shadow: 0 6px 16px rgba(37, 99, 235, 0.5);
      transform: translateY(-1px);
    }

    .action-btn:disabled {
      background: #334155;
      color: #94a3b8;
      cursor: not-allowed;
      box-shadow: none;
      transform: none;
    }

    .sample-link {
      display: block;
      margin-top: 12px;
      color: var(--accent-cyan);
      font-size: 0.82rem;
      cursor: pointer;
      text-decoration: underline;
    }

    .loading-banner {
      display: none;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid var(--accent-blue);
      border-radius: 8px;
      padding: 16px;
      margin-top: 20px;
      align-items: center;
      justify-content: center;
      gap: 12px;
    }

    .spinner {
      width: 22px;
      height: 22px;
      border: 3px solid rgba(56, 139, 253, 0.2);
      border-top-color: var(--accent-blue);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    #resultsSection {
      display: none;
      animation: fadeIn 0.4s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .section-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: #ffffff;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Grid for Summary */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }

    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 16px;
      text-align: center;
    }

    .stat-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 6px;
    }

    .stat-value {
      font-size: 1.7rem;
      font-weight: 800;
      color: #ffffff;
    }

    .stat-value.green { color: var(--accent-green); }
    .stat-value.red { color: var(--accent-red); }
    .stat-value.amber { color: var(--accent-amber); }
    .stat-value.cyan { color: var(--accent-cyan); }

    /* Highlights Split */
    .highlights-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }

    @media (max-width: 768px) {
      .highlights-grid { grid-template-columns: 1fr; }
    }

    .card-panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 20px;
    }

    .risk-badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      margin-left: 8px;
    }

    .risk-badge.high { background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #f85149; }
    .risk-badge.medium { background: rgba(240, 136, 62, 0.2); color: #f0883e; border: 1px solid #f0883e; }
    .risk-badge.low { background: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid #2ea043; }

    .reasons-list {
      list-style: none;
      margin-top: 12px;
    }

    .reasons-list li {
      padding: 6px 10px;
      background: #1a2233;
      border-left: 3px solid var(--accent-red);
      border-radius: 4px;
      font-size: 0.85rem;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    /* Video player container */
    .video-container {
      background: #000000;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--card-border);
      margin-bottom: 24px;
      text-align: center;
    }

    video {
      width: 100%;
      max-height: 520px;
      display: block;
      background: #000;
    }

    .evidence-bar {
      background: rgba(46, 160, 67, 0.15);
      border: 1px solid rgba(46, 160, 67, 0.35);
      color: #3fb950;
      padding: 12px 16px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.9rem;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }

    .evidence-gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }

    .evidence-thumb {
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid var(--card-border);
      cursor: pointer;
      transition: transform 0.2s;
      background: #000;
    }

    .evidence-thumb:hover {
      transform: scale(1.02);
      border-color: var(--accent-blue);
    }

    .evidence-thumb img {
      width: 100%;
      height: 120px;
      object-fit: cover;
      display: block;
    }

    .btn-secondary {
      background: #1e293b;
      border: 1px solid #334155;
      color: #cbd5e1;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
    }

    .btn-secondary:hover {
      background: #334155;
      color: #ffffff;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1 class="brand-title">VISIONX — WORKER SAFETY MONITOR</h1>
      <p class="brand-subtitle">AI-powered worker PPE compliance and contextual safety risk analysis.</p>
      <div class="badge-bar">
        <span class="badge live">&#9679; YOLOv8 + ByteTrack</span>
        <span class="badge">&#9679; Dynamic Risk Engine (0–100)</span>
        <span class="badge">&#9679; Scale-Adaptive Hazard Proximity</span>
        <span class="badge">&#9679; Automated Forensic Evidence</span>
      </div>
    </header>

    <main>
      <!-- Upload Card -->
      <section class="upload-card">
        <div class="drop-zone" id="dropZone">
          <div class="drop-icon">&#128249;</div>
          <p class="drop-text">Drag & drop construction video here, or click to browse</p>
          <p class="drop-subtext">Supported formats: MP4, AVI, MOV, MKV</p>
          <input type="file" id="fileInput" accept=".mp4,.avi,.mov,.mkv">
        </div>

        <div id="filePill" class="file-pill">
          <span>&#128196;</span>
          <span id="fileName">No file chosen</span>
        </div>

        <div style="margin-top: 10px;">
          <button id="analyzeBtn" class="action-btn" disabled>
            <span>&#9654;</span> ANALYZE VIDEO
          </button>
        </div>

        <span class="sample-link" id="sampleLink">Or analyze sample video (source_files/video_3.mp4)</span>

        <div id="loadingBanner" class="loading-banner">
          <div class="spinner"></div>
          <div>
            <strong>Analyzing video... Please wait.</strong>
            <div style="font-size: 0.8rem; color: var(--text-muted);">Executing YOLO PPE detection, worker tracking, hazard proximity & risk scoring...</div>
          </div>
        </div>
      </section>

      <!-- Results Section -->
      <section id="resultsSection">
        <!-- Site Summary -->
        <h2 class="section-title">&#128202; SITE SAFETY SUMMARY</h2>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-label">Total Workers</div>
            <div id="statTotalWorkers" class="stat-value">0</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Compliant Workers</div>
            <div id="statCompliant" class="stat-value green">0</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Violations Detected</div>
            <div id="statViolations" class="stat-value red">0</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Check / Uncertain</div>
            <div id="statCheck" class="stat-value amber">0</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Compliance Rate</div>
            <div id="statRate" class="stat-value cyan">0%</div>
          </div>
        </div>

        <!-- Highlights Grid -->
        <div class="highlights-grid">
          <!-- Highest Risk Worker -->
          <div class="card-panel">
            <h3 class="section-title" style="font-size: 1rem;">
              &#9888; HIGHEST RISK WORKER
              <span id="riskBadge" class="risk-badge low">LOW</span>
            </h3>
            <div style="margin: 12px 0;">
              <span id="riskWorkerId" style="font-weight: 700; font-size: 1.1rem;">Worker #None</span>
              <div style="font-size: 0.9rem; color: var(--text-muted); margin-top: 4px;">
                Risk Score: <strong id="riskScoreVal" style="color: #fff;">0</strong> / 100
              </div>
            </div>
            <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px;">
              Contributing Hazard Factors:
            </div>
            <ul id="reasonsList" class="reasons-list">
              <li>All safety gear in place</li>
            </ul>
          </div>

          <!-- Video Processing Telemetry -->
          <div class="card-panel">
            <h3 class="section-title" style="font-size: 1rem;">&#9881; PROCESSING TELEMETRY</h3>
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 10px; line-height: 1.8;">
              <div>Frames Analyzed: <strong id="teleFrames">0</strong></div>
              <div>Processing Time: <strong id="teleTime">0.0s</strong></div>
              <div>Execution Device: <strong>""" + str(DEVICE).upper() + """</strong></div>
              <div>Model: <strong>models/best.pt (10 classes)</strong></div>
              <div>Tracking Engine: <strong>ByteTrack (persist=True)</strong></div>
            </div>
          </div>
        </div>

        <!-- Evidence Alert Bar -->
        <div id="evidenceBar" class="evidence-bar">
          <div>
            &#10004; <strong>Safety evidence captured.</strong>
            <span id="evidenceCountText" style="margin-left: 8px; font-weight: normal; color: #cbd5e1;"></span>
          </div>
          <a href="/api/csv" class="btn-secondary" download>&#128196; Download safety_events.csv</a>
        </div>

        <!-- Annotated Video Player -->
        <div class="card-panel" style="margin-bottom: 24px;">
          <h2 class="section-title">&#127916; PROCESSED ANNOTATED VIDEO</h2>
          <div class="video-container">
            <video id="videoPlayer" controls>
              <source id="videoSource" src="" type="video/mp4">
              Your browser does not support the video tag.
            </video>
          </div>
          <div style="text-align: right;">
            <a id="downloadVideoBtn" href="#" class="btn-secondary" download>&#11015; Download Annotated Video</a>
          </div>
        </div>

        <!-- Captured Snapshots Gallery -->
        <div class="card-panel">
          <h2 class="section-title">&#128247; VIOLATION EVIDENCE SNAPSHOTS</h2>
          <div id="snapshotsGallery" class="evidence-gallery">
            <p style="color: var(--text-muted); font-size: 0.85rem;">No violation snapshots generated.</p>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const filePill = document.getElementById('filePill');
    const fileName = document.getElementById('fileName');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingBanner = document.getElementById('loadingBanner');
    const resultsSection = document.getElementById('resultsSection');
    const sampleLink = document.getElementById('sampleLink');

    let currentFile = null;

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        setFile(e.target.files[0]);
      }
    });

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        setFile(e.dataTransfer.files[0]);
      }
    });

    function setFile(file) {
      currentFile = file;
      fileName.textContent = file.name + ' (' + (file.size / (1024 * 1024)).toFixed(2) + ' MB)';
      filePill.style.display = 'inline-flex';
      analyzeBtn.disabled = false;
    }

    sampleLink.addEventListener('click', async () => {
      analyzeSample();
    });

    async function analyzeSample() {
      startLoading();
      try {
        const response = await fetch('/api/analyze-sample', { method: 'POST' });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        renderResults(data);
      } catch (err) {
        alert('Error processing sample video: ' + err.message);
      } finally {
        stopLoading();
      }
    }

    analyzeBtn.addEventListener('click', async () => {
      if (!currentFile) return;
      startLoading();

      const formData = new FormData();
      formData.append('file', currentFile);

      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          body: formData
        });
        if (!response.ok) {
          const errData = await response.json();
          throw new Error(errData.detail || 'Processing failed');
        }
        const data = await response.json();
        renderResults(data);
      } catch (err) {
        alert('Analysis Error: ' + err.message);
      } finally {
        stopLoading();
      }
    });

    function startLoading() {
      analyzeBtn.disabled = true;
      loadingBanner.style.display = 'flex';
      resultsSection.style.display = 'none';
    }

    function stopLoading() {
      analyzeBtn.disabled = false;
      loadingBanner.style.display = 'none';
    }

    function renderResults(data) {
      document.getElementById('statTotalWorkers').textContent = data.total_workers;
      document.getElementById('statCompliant').textContent = data.compliant_workers;
      document.getElementById('statViolations').textContent = data.violations_detected;
      document.getElementById('statCheck').textContent = data.check_workers;
      document.getElementById('statRate').textContent = data.compliance_rate + '%';

      document.getElementById('riskWorkerId').textContent = data.highest_risk_worker;
      document.getElementById('riskScoreVal').textContent = data.highest_risk_score;

      const riskBadge = document.getElementById('riskBadge');
      riskBadge.textContent = data.highest_risk_level;
      riskBadge.className = 'risk-badge ' + data.highest_risk_level.toLowerCase();

      const reasonsList = document.getElementById('reasonsList');
      reasonsList.innerHTML = '';
      data.detected_reasons.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        reasonsList.appendChild(li);
      });

      document.getElementById('teleFrames').textContent = data.frames_processed;
      document.getElementById('teleTime').textContent = data.processing_time_seconds + 's';

      const evidenceBar = document.getElementById('evidenceBar');
      if (data.safety_evidence_captured) {
        evidenceBar.style.display = 'flex';
        document.getElementById('evidenceCountText').textContent =
          `(${data.snapshots_generated} snapshots generated, CSV audit log updated)`;
      } else {
        evidenceBar.style.display = 'none';
      }

      // Video Player
      if (data.video_url) {
        const player = document.getElementById('videoPlayer');
        const source = document.getElementById('videoSource');
        source.src = data.video_url + '?t=' + Date.now();
        player.load();
        document.getElementById('downloadVideoBtn').href = data.download_url || data.video_url;
      }

      // Snapshots
      const gallery = document.getElementById('snapshotsGallery');
      gallery.innerHTML = '';
      if (data.snapshot_urls && data.snapshot_urls.length > 0) {
        data.snapshot_urls.forEach(url => {
          const div = document.createElement('div');
          div.className = 'evidence-thumb';
          div.innerHTML = `<img src="${url}" alt="Evidence Snapshot" onclick="window.open('${url}', '_blank')">`;
          gallery.appendChild(div);
        });
      } else {
        gallery.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">No violation snapshots triggered.</p>';
      }

      resultsSection.style.display = 'block';
      resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
  </script>
</body>
</html>
"""


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the VisionX Worker Safety Monitor browser application."""
    return HTMLResponse(content=HTML_CONTENT, status_code=200)


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Accepts an uploaded construction video file (MP4, AVI, MOV, MKV),
    processes it through the unified VisionXProcessor, and returns summary stats.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    allowed_exts = [".mp4", ".avi", ".mov", ".mkv"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: MP4, AVI, MOV, MKV."
        )

    # Generate safe unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"upload_{timestamp}{ext}"
    input_path = os.path.join(UPLOAD_DIR, safe_name)
    output_filename = f"annotated_{timestamp}.mp4"
    output_path = os.path.join(PROCESSED_DIR, output_filename)

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        file.file.close()

    # Process through unified processor
    try:
        summary = processor.process_video(input_path, output_path)
    except Exception as e:
        # Clean temporary file
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")

    # Remove temporary upload file to conserve disk space
    if os.path.exists(input_path):
        os.remove(input_path)

    # Format snapshot URLs for frontend display
    snapshot_urls = [
        f"/snapshots/{os.path.basename(p)}" for p in summary.get("snapshot_paths", [])
    ]

    return JSONResponse({
        **summary,
        "video_url": f"/processed/{output_filename}",
        "download_url": f"/download/{output_filename}",
        "snapshot_urls": snapshot_urls,
    })


@app.post("/api/analyze-sample")
async def analyze_sample_video():
    """
    Convenience endpoint for hackathon demo:
    Quickly analyzes the bundled sample video source_files/hardhat.mp4 (first 100 frames for speed).
    """
    sample_path = os.path.join("source_files", "video_3.mp4")
    if not os.path.exists(sample_path):
        sample_path = os.path.join("source_files", "hardhat.mp4")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample video not found.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"sample_annotated_{timestamp}.mp4"
    output_path = os.path.join(PROCESSED_DIR, output_filename)

    try:
        # Process first 100 frames of the sample video for fast hackathon response
        summary = processor.process_video(sample_path, output_path, max_frames=100)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample processing failed: {e}")

    snapshot_urls = [
        f"/snapshots/{os.path.basename(p)}" for p in summary.get("snapshot_paths", [])
    ]

    return JSONResponse({
        **summary,
        "video_url": f"/processed/{output_filename}",
        "download_url": f"/download/{output_filename}",
        "snapshot_urls": snapshot_urls,
    })


@app.get("/processed/{filename}")
async def get_processed_video(filename: str):
    """Serves the generated annotated video file."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Processed video not found.")
    return FileResponse(path, media_type="video/mp4")


@app.get("/download/{filename}")
async def download_processed_video(filename: str):
    """Downloads the generated annotated video file."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, filename=filename, media_type="application/octet-stream")


@app.get("/snapshots/{filename}")
async def get_snapshot(filename: str):
    """Serves an evidence snapshot image."""
    path = os.path.join(SNAPSHOTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/csv")
async def download_csv_log():
    """Downloads the persistent safety events CSV log."""
    csv_path = os.path.join(LOGS_DIR, "safety_events.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV log not found.")
    return FileResponse(csv_path, filename="safety_events.csv", media_type="text/csv")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import socket

    HOST = "127.0.0.1"
    PORT = 8000

    print()
    print("=" * 60)
    print("VISIONX - WORKER SAFETY MONITOR WEB SERVER")
    print(f"Local Server URL: http://{HOST}:{PORT}")

    # Check if port is already active
    def is_port_in_use(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0

    if is_port_in_use(HOST, PORT):
        print(f"[VisionX] Port {PORT} is already in use by an active VisionX server.")
        print(f"[VisionX] Dashboard is currently accessible at: http://{HOST}:{PORT}")
        print("=" * 60)
        print()
    else:
        print("Open your browser and navigate to the above URL.")
        print("Press Ctrl+C to stop the server.")
        print("=" * 60)
        print()

        try:
            uvicorn.run(app, host=HOST, port=PORT, log_level="info")
        except OSError as e:
            if getattr(e, "winerror", None) == 10048 or "10048" in str(e):
                print(f"[VisionX] Port {PORT} is already occupied. Dashboard is running at http://{HOST}:{PORT}")
            else:
                raise
