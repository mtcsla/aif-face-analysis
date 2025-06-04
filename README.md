# aif-data-analysis

Real-Time Face Analysis Web Application

---

## Face Analysis Logic

The core face analysis logic is implemented in `src/logic/face_analysis.py` and is used by the FastAPI backend to process both live video frames and uploaded images. The main steps are:

- **Frame Preprocessing:**  
  Each frame is resized to a smaller width (default: 320px) while preserving aspect ratio. This reduces computational load and speeds up analysis. The scaling factors are saved to map results back to the original frame size.

- **Emotion Analysis (with Caching):**  
  Emotion is analyzed using DeepFace every N frames (default: 5) for live video, or every frame for static images. The dominant emotion and face region are cached and reused between analyses to improve performance.

- **Age and Gender Analysis (with Throttling):**  
  Age and gender are analyzed using DeepFace only once per second for live video, with results cached and reused for subsequent frames. This balances accuracy and efficiency.

- **Label Composition:**  
  The detected gender, age, and emotion are combined into a single label (e.g., "Woman, 25, happy") for display above the detected face.

- **Error Handling:**  
  If DeepFace fails to detect a face or returns an error, the system gracefully falls back to displaying only the available information.

- **Drawing and Color Coding:**  
  Detected faces are highlighted with rounded rectangles, and the color of the rectangle is determined by the detected emotion (e.g., green for happy, red for angry).

This modular approach allows for efficient, real-time face analysis and annotation, making the application responsive even under continuous video streaming.

---

## Overview

aif-data-analysis is a real-time face analysis web application that leverages deep learning to detect faces, estimate age and gender, and analyze emotions from webcam streams or uploaded images. The project features a modern FastAPI backend, a responsive frontend built with DaisyUI/Tailwind CSS, and is designed for robust, secure, and scalable deployment on Kubernetes with Traefik handling TLS and WebSocket streaming.

---

## Features

- **Real-Time Face Analysis**: Detects faces, estimates age/gender, and recognizes emotions from live webcam streams or uploaded images.
- **Modern Web UI**: Responsive interface using DaisyUI and Tailwind CSS, with live video/canvas display and image upload.
- **WebSocket Streaming**: Low-latency video frame streaming and annotation via WebSockets.
- **Secure by Design**: TLS termination and secure routing via Traefik; authentication required for HTML page access.
- **Kubernetes Native**: All deployment manifests provided (Deployment, Service, IngressRoute, Secrets).
- **Dependency Transparency**: Modal dialog in the UI lists all Python dependencies.
- **GitHub Integration**: Direct link to the project repository from the web UI.

---

## Tech Stack

- **Backend**: Python, FastAPI, WebSockets, OpenCV, DeepFace, Starlette
- **Frontend**: HTML, DaisyUI, Tailwind CSS, Vanilla JS
- **Containerization**: Docker, GitHub Container Registry (GHCR)
- **Orchestration**: Kubernetes (manifests in `kubernetes/`)
- **Ingress & TLS**: Traefik (IngressRoute, Middleware, Let's Encrypt)
- **Secrets Management**: Kubernetes Secrets for app credentials and GHCR access

---

## Directory Structure

```
aif-face-analysis/
├── src/                  # Backend FastAPI application
│   └── server.py
├── resources/            # Frontend static files
│   └── index.html
├── kubernetes/           # Kubernetes manifests (deployment, service, ingress, secrets)
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── ghcr-secret.yaml
│   └── app-secret.yaml
├── requirements.txt      # Python dependencies
├── README.md             # This file
```

---

## Setup & Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/mtcsla/aif-face-analysis.git
cd aif-face-analysis/aif-data-analysis
```

### 2. Install Python Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file (or use Kubernetes secrets in production):

```
APP_USERNAME=your_username
APP_PASSWORD=your_password
```

### 4. Run Locally

```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
```

Visit [http://localhost:8000](http://localhost:8000) and log in with your credentials.

---

## Kubernetes Deployment

> **Note:**  
> The provided Kubernetes manifests assume you have a Traefik instance installed in your cluster with the ingress class name `traefik`, and that all Traefik Custom Resource Definitions (CRDs) from `traefik.io/v1alpha1` are present.  
> See the [Traefik documentation](https://doc.traefik.io/traefik/getting-started/install-traefik/) for installation instructions.

1. **Build and Push Docker Image** (if modifying backend):

   ```bash
   docker build -t ghcr.io/mtcsla/aif-face-analysis/face-analysis-app:latest .
   docker push ghcr.io/mtcsla/aif-face-analysis/face-analysis-app:latest
   ```

2. **Set up Secrets**:

   - Create `ghcr-secret.yaml` with your GitHub Container Registry credentials (see file for envsubst instructions).
   - Create `app-secret.yaml` with your app credentials (base64-encoded).

3. **Apply Manifests**:

   ```bash
   kubectl apply -f kubernetes/ghcr-secret.yaml
   kubectl apply -f kubernetes/app-secret.yaml
   kubectl apply -f kubernetes/service.yaml
   kubectl apply -f kubernetes/deployment.yaml
   kubectl apply -f kubernetes/ingress.yaml
   ```

4. **Access the App**:

   - The app will be available at `https://aif.mtcsla.dev` (or your configured domain).

---

## Usage

- **Live Stream**: Click "Start Stream" to analyze your webcam feed in real time.
- **Image Upload**: Upload an image to analyze faces, age, gender, and emotion.
- **Dependencies**: Click "Show Dependencies" to view all Python packages used.
- **GitHub**: Use the GitHub button in the UI to visit the repository.

---

## License

MIT License

---

## Credits

- Developed by Mateusz Cieśla, Esbol Erlan, Sheaba El-Esawy, Rubayet Rafsan

---

## Links

- [Live Demo](https://aif.mtcsla.dev)
- [GitHub Repository](https://github.com/mtcsla/aif-face-analysis)
