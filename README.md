# Face smile AI

Detect **smiles** on faces using **OpenCV Haar Cascades**, wrapped in a clean **Streamlit** UI.  
Supports **Realtime via WebRTC** (live webcam) and **Snapshot/Upload** (take a photo or upload an image).

---

## Table of Contents
- [Features](#features)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [How It Works](#how-it-works)
- [App Usage](#app-usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Privacy](#privacy)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## ✨ Features

- 🔴 **Realtime (WebRTC)**: Process frames directly from your webcam; draw face boxes and a `Smiling :D` label.
- 📷 **Snapshot/Upload**: Use the browser camera to take a photo or upload a local image for detection.
- ⚙️ **Tunable parameters** in the sidebar: `scaleFactor`, `minNeighbors`, minimum face size, FPS overlay, etc.
- 🧱 **Zero extra assets**: Haar cascade XMLs are loaded from `cv2.data.haarcascades` (bundled with OpenCV).
- 🪶 Lightweight and simple—great for demos, labs, or quick proofs-of-concept.

---

## 🖼️ Screenshots

_(Optional: add GIFs or images here)_

- Realtime WebRTC detection with FPS overlay.
- Snapshot/Upload results with bounding boxes and labels.

---

## 🧰 Requirements

- **Python**: 3.8+
- **OS**: Windows / macOS / Linux
- **Browser**: Modern browser with WebRTC support (Chrome/Edge/Firefox/Safari)

---

## 🚀 Quickstart

> Using a virtual environment is recommended for clean, reproducible installs.

### 1) Create & activate a virtual environment

**Venv (standard Python)**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
