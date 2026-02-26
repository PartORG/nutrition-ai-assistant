# Nutrition AI Assistant

A personalized nutrition and recipe recommendation system powered by LLM, RAG, and computer vision.

The system understands user health conditions, dietary restrictions, and daily nutrition budgets. It recommends safe recipes, detects food ingredients from photos, tracks daily nutrition, and holds natural conversations through a chat interface.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Flutter App (iOS / Android / Windows / macOS / Linux / Web)│
│  Cross-platform client — calls REST API + WebSocket         │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTP / WebSocket  (port 8000)
┌─────────────────▼───────────────────────────────────────────┐
│  FastAPI Backend  (src/)                                    │
│  • LangChain agent with 8 tools                             │
│  • Medical RAG + Recipe RAG (FAISS + BM25 hybrid)          │
│  • Safety filter, intent parser                             │
│  • SQLite: users, recipes, nutrition, chat history          │
└────────────┬────────────────────────┬───────────────────────┘
             │                        │
  ┌──────────▼──────────┐   ┌─────────▼──────────┐
  │  Ollama  (port 11434)│   │ YOLO Detector       │
  │  • llama3.2 (RAG)   │   │   (port 8001)       │
  │  • llava (fallback) │   │ • YOLOv8 detection  │
  │  • embeddings       │   │ • Food101 ResNet18  │
  └─────────────────────┘   │ • isolated env      │
                             └─────────────────────┘
```

**Agent tools:** `search_recipes` · `save_recipe` · `show_recipe` · `analyze_image` · `nutrition_status` · `general_chat` · `safety_guard` · `crisis_support`

**Image detection:** YOLO microservice (primary) → LLaVA via Ollama (fallback if YOLO finds nothing or is unavailable)

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Backend & YOLO service |
| Docker + Docker Compose | latest | Containerized stack |
| Ollama | latest | Local LLMs (llama3.2, llava) |
| Flutter SDK | 3.x | Mobile/desktop app |

---

## Option A — Docker (recommended)

Runs all three backend services in containers. Flutter connects to `http://localhost:8000`.

### 1. Place the YOLO model weights

```
services/yolo_detector/models/food101_resnet18_best.pth
```

`yolov8n.pt` is downloaded automatically by ultralytics on first run.

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set:

```ini
JWT_SECRET=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
GROQ_API_KEY=gsk_...          # from console.groq.com/keys
```

Everything else works with defaults for Docker.

### 3. Start the stack

```bash
docker-compose up --build
```

### 4. Pull Ollama models (first run only)

```bash
docker exec -it nutriai-ollama ollama pull llama3.2
docker exec -it nutriai-ollama ollama pull llava
```

### 5. Verify

```
http://localhost:8000/health        → {"status": "ok"}
http://localhost:8001/health        → {"status": "ok", "model_loaded": true}
http://localhost:11434              → Ollama running
```

### Stop

```bash
docker-compose down
```

---

## Option B — Local development (no Docker)

Run each component in its own terminal. Good for debugging with hot-reload.

### Terminal 1 — Ollama

```bash
ollama serve
ollama pull llama3.2
ollama pull llava
```

### Terminal 2 — YOLO microservice (isolated venv)

```bash
cd services/yolo_detector
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --port 8001
```

Verify: `curl http://localhost:8001/health`

### Terminal 3 — FastAPI backend

```bash
# From project root, activate your main venv
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements-dev.txt

python run_api.py
```

API is live at `http://localhost:8000`.

### Flutter app

```bash
cd app/nutrition_ai_assistent
flutter pub get
flutter run
```

Point the app to `http://localhost:8000` (already the default).

---

## Project structure

```
nutrition-ai-assistent/
├── app/                        # Flutter cross-platform client
│   └── nutrition_ai_assistent/
├── src/                        # FastAPI backend
│   ├── adapters/rest/          # HTTP routes, WebSocket chat
│   ├── agent/                  # LangChain agent + 8 tools
│   ├── application/            # Services, DTOs, context
│   ├── domain/                 # Models, ports, exceptions
│   └── infrastructure/         # RAG, LLM, CNN adapters, SQLite repos
├── services/
│   └── yolo_detector/          # YOLO microservice (isolated Python env)
│       ├── detector.py         # YOLOv8 + Food101 ResNet18 pipeline
│       ├── main.py             # FastAPI wrapper (POST /detect)
│       ├── requirements.txt    # ultralytics, opencv-headless, torch
│       ├── Dockerfile
│       └── models/             # Place food101_resnet18_best.pth here
├── data/                       # Recipe + nutrition datasets (gitignored)
├── data_test/                  # Medical PDF documents (gitignored)
├── vector_databases/           # FAISS vector stores (gitignored)
├── Dockerfile                  # Main API container
├── docker-compose.yml          # Wires all 3 containers
├── run_api.py                  # Local dev entry point
├── .env.example                # Environment variable template
└── requirements-dev.txt        # Main app Python dependencies
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET` | *(required)* | Secret key for JWT tokens |
| `JWT_EXPIRY_HOURS` | `24` | Token lifetime |
| `DB_PATH` | `users.db` | SQLite database path |
| `AGENT_LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `AGENT_LLM_MODEL` | `llama-3.3-70b-versatile` | Agent LLM model name |
| `GROQ_API_KEY` | *(required for groq)* | Groq cloud API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434/` | Ollama server URL |
| `LLM_MODEL` | `llama3.2` | RAG / intent parser model |
| `CNN_DETECTOR_TYPE` | `yolo_with_fallback` | `yolo_with_fallback` / `yolo_only` / `llava_only` |
| `YOLO_SERVICE_URL` | `http://localhost:8001` | YOLO microservice URL |
| `CNN_MODEL_PATH` | `llava` | LLaVA model name (fallback detector) |

---

## Data setup

The RAG systems require pre-built vector databases and source data. Place them in:

```
data/                  # Recipe and nutrition CSV/JSON files
data_test/raw/         # Medical PDF documents
vector_databases/      # Pre-built FAISS indexes
```

These directories are gitignored due to size. Contact the team or re-build the indexes by running the ingestion scripts in `notebooks/`.

---

## Detector modes

| Mode | Behavior |
|---|---|
| `yolo_with_fallback` | YOLO runs first. If it finds no ingredients (image has no recognizable COCO food items) or the service is down, falls back to LLaVA automatically. |
| `yolo_only` | YOLO only. Fast, structured output. Works best with whole foods (banana, pizza, broccoli, etc.). |
| `llava_only` | LLaVA via Ollama only. Free-form ingredient recognition — handles any ingredient including raw ingredients, packaged foods, and complex dishes. |
