# Face smile AI

Detect **smiles** on faces using **OpenCV Haar Cascades**, wrapped in a clean **Streamlit** UI.  
Supports **Realtime via WebRTC** (live webcam) and **Snapshot/Upload** (take a photo or upload an image).

## Features

- 🔴 **Realtime (WebRTC)**: Process frames directly from your webcam; draw face boxes and a `Smiling :D` label.
- 📷 **Snapshot/Upload**: Use the browser camera to take a photo or upload a local image for detection.
- ⚙️ **Tunable parameters** in the sidebar: `scaleFactor`, `minNeighbors`, minimum face size, FPS overlay, etc.
- 🧱 **Zero extra assets**: Haar cascade XMLs are loaded from `cv2.data.haarcascades` (bundled with OpenCV).
- 🪶 Lightweight and simple—great for demos, labs, or quick proofs-of-concept.

## Demos:

<img width="1919" height="990" alt="image" src="https://github.com/user-attachments/assets/65c6c208-5d6a-4fed-83ac-f00a5ccfb196" />
<img width="1919" height="998" alt="image" src="https://github.com/user-attachments/assets/e980ff72-9216-43c0-af92-b73026276ef0" />
https://github.com/user-attachments/assets/3f6a7b26-c92b-44e7-8cd0-f13cc2988e81


## How to install:

> Using a virtual environment is recommended for clean, reproducible installs.

### 1) Create & activate a virtual environment

**Venv (standard Python)**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
