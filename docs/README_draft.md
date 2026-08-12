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
└────────────┬────────────────────────────────────────────────┘
             │
  ┌──────────▼──────────┐
  │  Ollama  (port 11434)│
  │  • llama3.2 (RAG)   │
  │  • llava (vision)   │
  │  • embeddings       │
  └─────────────────────┘
```

**Agent tools:** `search_recipes` · `save_recipe` · `show_recipe` · `analyze_image` · `nutrition_status` · `general_chat` · `safety_guard` · `crisis_support`

**Image detection:** LLaVA via Ollama — a vision-language model that extracts ingredients from food photos.

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Backend |
| Docker + Docker Compose | latest | Containerized stack |
| Ollama | latest | Local LLMs (llama3.2, llava) |
| Flutter SDK | 3.x | Mobile/desktop app |

---

## Option A — Docker (recommended)

Runs the backend services in containers. Flutter connects to `http://localhost:8000`.

### 1. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set:

```ini
JWT_SECRET=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
GROQ_API_KEY=gsk_...          # from console.groq.com/keys
```

Everything else works with defaults for Docker.

### 2. Start the stack

```bash
docker-compose up --build
```

### 3. Pull Ollama models (first run only)

```bash
docker exec -it nutriai-ollama ollama pull llama3.2
docker exec -it nutriai-ollama ollama pull llava
```

### 4. Verify

```
http://localhost:8000/health        → {"status": "ok"}
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

### Terminal 2 — FastAPI backend

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
├── data/                       # Recipe + nutrition datasets (gitignored)
├── data_test/                  # Medical PDF documents (gitignored)
├── vector_databases/           # FAISS vector stores (gitignored)
├── Dockerfile                  # Main API container
├── docker-compose.yml          # Wires the API + Ollama containers
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
| `CNN_MODEL_PATH` | `llava` | LLaVA model name for image ingredient detection |

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

## Image ingredient detection

Food photos are analyzed by **LLaVA** (a vision-language model) via Ollama. It performs
free-form ingredient recognition — handling any ingredient including raw ingredients,
packaged foods, and complex dishes — and returns a list of detected ingredients.
