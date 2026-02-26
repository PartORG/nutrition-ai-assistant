# Architecture Guide: Nutrition AI Assistant

> **Audience:** Junior to mid-level developers joining this project.
> **Goal:** Explain *why* this architecture was chosen, *how* every piece fits together, and *what to change* when you need to extend the system.

---

## Table of Contents

0. [For Non-Programmers: Plain English Guide](#0-for-non-programmers-plain-english-guide)
1. [The Big Picture](#1-the-big-picture)
2. [Why Hexagonal Architecture?](#2-why-hexagonal-architecture)
3. [Folder Structure Explained](#3-folder-structure-explained)
4. [Layer-by-Layer Walkthrough](#4-layer-by-layer-walkthrough)
   - 4.1 [Domain — The Heart of the System](#41-domain--the-heart-of-the-system)
   - 4.2 [Application — Use Cases & Orchestration](#42-application--use-cases--orchestration)
   - 4.3 [Infrastructure — The Outside World](#43-infrastructure--the-outside-world)
   - 4.4 [Agent — The Conversational Brain](#44-agent--the-conversational-brain)
   - 4.5 [Adapters — Entry Points](#45-adapters--entry-points)
   - 4.6 [Factory — The Composition Root](#46-factory--the-composition-root)
5. [Complete Request Flow](#5-complete-request-flow)
   - 5.1 [Chat Message (WebSocket)](#51-chat-message-websocket)
   - 5.2 [Recipe Recommendation Pipeline](#52-recipe-recommendation-pipeline)
   - 5.3 [Image Analysis Flow](#53-image-analysis-flow)
6. [The Flutter App](#6-the-flutter-app)
7. [Docker Deployment](#7-docker-deployment)
8. [How to: Add a New Tool](#8-how-to-add-a-new-tool)
9. [How to: Change the Database](#9-how-to-change-the-database)
10. [How to: Change the Agent Prompt](#10-how-to-change-the-agent-prompt)
11. [How to: Change the RAG Prompt](#11-how-to-change-the-rag-prompt)
12. [How to: Switch or Change the LLM Model](#12-how-to-switch-or-change-the-llm-model)
13. [Configuration Reference](#13-configuration-reference)
14. [Functional Testing Scripts](#14-functional-testing-scripts)

---

## 0. For Non-Programmers: Plain English Guide

> **Who is this section for?** Product managers, designers, project stakeholders, or anyone curious about how this system works — without needing to read a single line of code.

---

### 0.1 What does this app actually do?

Imagine a personal nutritionist who lives in your phone. You can type to them — or even take a photo of your fridge — and they will:

- Suggest personalized recipes based on your health conditions (diabetes, hypertension, etc.)
- Make sure every recipe is safe given your allergies and dietary restrictions
- Remember your conversation from last time
- Track what you've already eaten today and adjust suggestions accordingly
- Let you save a recipe to your history with a tap

That nutritionist is this software. The "brain" is a large language model (the same technology behind ChatGPT), guided by a structured knowledge base of medical documents and recipe datasets.

---

### 0.2 The two parts: App and Server

The system has two halves that talk to each other:

```
┌────────────────────────┐        ┌────────────────────────────────┐
│   Flutter Mobile App   │ ←────→ │   Python Backend Server        │
│                        │        │                                │
│  What users see and    │        │  Where all the intelligence    │
│  interact with.        │        │  lives: AI, recipes, database. │
│  (your phone screen)   │        │  (runs on a server/computer)   │
└────────────────────────┘        └────────────────────────────────┘
```

The **app** is just a window. It shows you text, buttons, and images. When you type a message, the app sends it to the server and waits for a reply — like texting someone who is very smart and very fast.

The **server** does all the thinking. It reads your message, decides what you want, looks up recipes, checks them for safety, and sends back a nicely formatted response.

---

### 0.3 What happens when you type a message?

Let's say you type: *"I have diabetes, can you suggest a low-sugar dinner?"*

Here is what happens behind the scenes, step by step:

**Step 1 — Your message travels to the server.**
The app sends your text over the internet to the server (this happens in under a second).

**Step 2 — A small AI reads your message.**
A specialized AI called the *Intent Parser* reads your message and extracts the important facts:
- You have diabetes
- You want a low-sugar meal
- It's for dinner

Think of it like a very smart secretary who reads your note and highlights the key points before passing it on.

**Step 3 — The server looks up your medical needs.**
The system checks a library of real medical documents (PDFs about dietary guidelines for various health conditions). From these, it figures out exactly what limits to apply — for example: "diabetic patients should have no more than 25g of sugar per meal."

It also remembers if you've already eaten today and adjusts the limits for the remaining daily budget.

**Step 4 — The system searches a recipe library.**
A second AI system searches through tens of thousands of recipes and nutrition facts to find the ones that best match your request and health needs.

**Step 5 — A safety checker reviews every recipe.**
Before any recipe is shown to you, a third AI checks: "Does this recipe contain anything this person must avoid? Does it exceed their sugar limit? Is it appropriate for their condition?" Unsafe recipes are quietly filtered out.

**Step 6 — The answer comes back to your screen.**
The formatted list of recipes appears in the app, complete with ingredients, instructions, and nutrition facts.

All of this happens in a few seconds.

---

### 0.4 Why is it built in separate pieces?

Think of it like a well-run restaurant versus a one-person street food stall.

**The street food stall (monolith — what we did NOT do):**

One person cooks, takes orders, handles money, washes dishes, and manages inventory — all at once. This works fine when there's only one customer. But what happens when 50 people arrive? Everything slows down. If the cook gets sick, the whole operation stops. If you want to change the menu, you have to retrain the same person on everything, not just cooking.

**The restaurant (what we built):**

There is a host who greets customers, waiters who take orders, a head chef who plans dishes, line cooks who specialize in sections (grill, salad, pastry), a cashier, and a dishwasher. Each person has one clear job. When it's busy, you hire more line cooks — you don't need to clone the whole restaurant. If the pastry chef leaves, you hire a new one and they slot right in without disrupting the rest of the kitchen.

This software works the same way. Each part has one clear job, and they communicate through well-defined "interfaces" — like tickets passing between the waiter and the kitchen.

---

### 0.5 The folders explained without jargon

The `src/` folder contains all the server code. Here is what each subfolder does in plain language:

```
src/
│
├── domain/          🧠 "The rules of the game"
│                    Contains the definitions of what things ARE.
│                    What is a User? What is a Recipe? What does a safety check look like?
│                    This never changes when you switch databases or AI providers.
│                    No external tools here — just pure definitions.
│
├── application/     🎯 "What the system knows how to DO"
│                    Contains the step-by-step workflows (use cases).
│                    "Here is how we find a recipe for a user."
│                    "Here is how we register a new account."
│                    "Here is how we save a recipe to history."
│                    It uses the rules from domain/ and asks infrastructure/ to do
│                    the heavy lifting (database, AI calls).
│
├── infrastructure/  🔧 "The actual machines and tools"
│                    The real database code. The real AI model connections.
│                    The YOLO computer vision service. The medical PDF reader.
│                    If you want to swap SQLite for PostgreSQL, or Ollama for ChatGPT,
│                    you change files ONLY in this folder.
│
├── agent/           🤖 "The conversational brain"
│                    The AI assistant that reads your chat messages, decides what
│                    action to take (search recipes? save a recipe? analyze a photo?),
│                    and remembers your conversation history.
│                    It has a collection of "tools" — like apps on a phone —
│                    each one doing a specific job.
│
├── adapters/        🚪 "The front doors"
│                    The code that listens for incoming requests from the Flutter app.
│                    It handles the WebSocket (live chat), the REST API (standard requests),
│                    and the authentication (checking who you are via your login token).
│                    It translates between "internet format" (JSON) and the internal
│                    world of the application.
│
└── factory.py       🏗️ "The assembly line foreman"
                     One single file that wires everything together at startup.
                     "The recommendation service needs the recipe AI, the medical AI,
                     the safety checker, and the database — here, take all of them."
                     This is the ONLY place that knows which concrete tool implements
                     which job. Everything else just asks for what it needs.
```

And the `app/` folder contains the Flutter mobile app — the user interface running on your phone.

---

### 0.6 Why is this better than putting it all in one file?

Here are five real situations where this structure saves the day:

---

**Situation 1: "We want to switch from local AI to ChatGPT."**

With a monolith (everything in one file), the AI model is woven into every part of the code. You'd have to find and rewrite dozens of places.

With this architecture, you open `factory.py`, change two lines that say which AI provider to use, and set an environment variable. Done. The rest of the code never knew which AI it was talking to — it just knew it could ask for recipe recommendations and get a structured answer back.

---

**Situation 2: "Two users are chatting at the same time."**

With a monolith using global variables, User A's conversation history might accidentally mix with User B's. This is a real, classic bug.

With this architecture, each WebSocket connection creates its own isolated "session context" — like giving each customer their own table, their own waiter, and their own order ticket. They cannot interfere with each other.

---

**Situation 3: "We need to add a new feature — show the user their meal history."**

With a monolith, you'd search through thousands of lines to find the right places and risk breaking something unrelated.

With this architecture, you add one new tool file in `agent/tools/`, register it in one line in `factory.py`, and the rest of the system picks it up automatically. The agent's instructions update themselves to mention the new tool.

---

**Situation 4: "The app is slow because the AI takes 3 seconds to respond."**

With a monolith using synchronous code, while the AI is thinking, the entire server is frozen — nobody else can get a response.

With this architecture, every AI call is "asynchronous" — while the AI is thinking for User A, the server is also serving User B, User C, and User D at the same time. It's like a chef who puts a dish in the oven and then starts the next dish, rather than standing and watching it cook.

---

**Situation 5: "A developer wants to test the recipe-saving logic."**

With a monolith, to test recipe saving you'd need a running database, a running AI model, and a running server.

With this architecture, the `SaveRecipeTool` only knows about a `RecipeManagerService`, and `RecipeManagerService` only knows about a repository interface. In a test, you replace the real database with a fake one (a "mock") that just pretends to save. The test runs instantly, with no real database needed.

---

### 0.7 How the whole thing works together — a visual story

```
You open the app and type: "I have eggs, spinach, and cheese. What can I make?"

         [Your Phone]
              │
              │  "I have eggs, spinach, and cheese. What can I make?"
              │
              ▼
   ┌─────────────────────┐
   │   Front Door        │  Checks your login token. Creates your session.
   │   (adapters/)       │  Loads your health profile from the database.
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   The Brain         │  Reads your message.
   │   (agent/)          │  Decides: "This is a recipe request."
   │                     │  Calls the search_recipes tool.
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   The Workflow      │  Runs the 5-step pipeline:
   │   (application/)    │  1. Parse intent → "has eggs, spinach, cheese"
   │                     │  2. Get medical limits → "no restrictions found"
   │                     │  3. Build enriched search query
   │                     │  4. Search recipe library
   │                     │  5. Safety check each recipe
   └──────────┬──────────┘
              │
    ┌─────────┴──────────────────────┐
    │                                │
    ▼                                ▼
 ┌──────────────┐          ┌──────────────────┐
 │  The Library │          │  The Database    │
 │  (infra/rag) │          │  (infra/persist) │
 │              │          │                  │
 │  Searches    │          │  Checks what     │
 │  100,000s of │          │  you've eaten    │
 │  recipes for │          │  today already.  │
 │  best match. │          │                  │
 └──────┬───────┘          └────────┬─────────┘
        │                           │
        └──────────┬────────────────┘
                   │
                   ▼
   ┌─────────────────────┐
   │   Safety Checker    │  "Does this recipe have anything
   │   (infra/llm)       │   you must avoid? Too much sodium?"
   │                     │  Removes unsafe recipes.
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   The Brain again   │  Formats the final answer.
   │   (agent/)          │  Saves your message + the reply to the database.
   └──────────┬──────────┘
              │
              ▼
         [Your Phone]

   "Here are 3 recipes using eggs, spinach, and cheese:
    1. Spinach and Feta Omelette (320 kcal)...
    2. Crustless Quiche (280 kcal)...
    3. Egg and Spinach Scramble (240 kcal)..."
```

Each box in that diagram is a separate, independent part of the code. They only talk to each other through agreed interfaces — like departments communicating through formal memos rather than one person running around doing everything themselves.

---

### 0.8 The "what if" test — how good design handles change

| What if we want to... | With a monolith | With this architecture |
|---|---|---|
| Switch the AI from local to cloud | Weeks of rewriting | Change 2 lines in `factory.py` + set an env variable |
| Move from SQLite to PostgreSQL | Risky refactor touching hundreds of lines | Write new repo files in `infrastructure/`, change `factory.py` |
| Add a new chat feature | Find the right place in thousands of lines, risk breaking things | Add one file in `agent/tools/`, one line in `factory.py` |
| Support 1,000 concurrent users | Server freezes on each AI call | Already async — each user's session is isolated |
| Test just the recipe-saving logic | Must run entire server with all services | Mock the database, test instantly in isolation |
| Update the AI's instructions | Buried in a 370-line inline string | One clear file: `agent/prompt.py` |
| Change what medical constraints are extracted | Tangled with recipe logic | One isolated class: `infrastructure/rag/medical_rag.py` |

---

*End of non-programmer section. The rest of this document goes deeper into the technical implementation.*

---

## 1. The Big Picture

This is an AI-powered nutrition assistant. Users chat with it (via a Flutter mobile app or any WebSocket client) and the system:

- Understands what the user is asking (Intent Parsing with LLM)
- Looks up their medical dietary constraints (Medical RAG from PDFs)
- Finds relevant recipes (Recipe RAG from CSV datasets)
- Filters recipes for safety against medical constraints (Safety Filter with LLM)
- Saves chosen recipes and tracks daily nutrition (SQLite database)
- Analyzes food photos to identify ingredients (YOLO + LLaVA vision models)

```
Flutter App
    │  REST + WebSocket
    ▼
FastAPI Server (src/)
    │
    ├── Auth → JWT tokens
    ├── WebSocket /ws/chat → Agent (LLM + Tools)
    │       ├── search_recipes → RAG pipeline
    │       ├── save_recipe → SQLite
    │       ├── analyze_image → YOLO / LLaVA
    │       ├── nutrition_status → SQLite
    │       └── ... more tools
    │
    ├── Ollama (local LLM: llama3.2, llava)
    ├── YOLO Detector (separate Docker container)
    └── SQLite Database (users.db)
```

---

## 2. Why Hexagonal Architecture?

### 2.1 Start here: what problem are we actually solving?

Before any architecture is chosen, there is always an underlying problem it is trying to solve. For this project the problem is:

> *"We have a complex AI system that talks to databases, multiple LLM providers, a computer-vision service, and a mobile app — and every single one of those things might change or be replaced tomorrow. How do we build this so that changing one piece never breaks another?"*

The answer is Hexagonal Architecture. But to understand *why* it answers that question, it helps to first see clearly what the alternative looks like and why it fails.

---

### 2.2 The Monolith: what we deliberately avoided

A monolith is not a bad word — it just means "everything is directly connected to everything else." For a small script or a weekend project that is perfectly fine. For a production AI system that multiple developers will extend over time, it creates a specific class of pain.

Here is what the monolith version of this project would have looked like:

```
agent.py  (1 500+ lines)
│
├── creates the database connection at the top of the file
├── creates the LLM client at the top of the file
├── creates the RAG system at the top of the file
├── creates the CNN detector at the top of the file
│
├── def run(user_input):
│       intent = ollama.chat(...)          ← directly calls Ollama
│       constraints = faiss.search(...)    ← directly uses FAISS
│       recipes = rag.ask(...)             ← calls RAG inline
│       db.execute("INSERT ...")           ← writes to DB inline
│       return recipes
│
└── (all prompts are long inline strings at line 200, 450, 800...)
```

Now imagine the following realistic requests:

---

**Request A: "Let's use Groq instead of Ollama — it's 10× faster."**

In the monolith, `ollama.chat(...)` appears in dozens of places: the intent parser, the safety filter, the RAG system, the agent loop. You have to find every one, understand the context, and rewrite them — while hoping you don't break something nearby.

In this project: change two environment variables. Done. Nothing else touches the LLM directly.

---

**Request B: "Two users are chatting at the same time and they're seeing each other's recipes."**

In the monolith, the state (current user, current conversation, last recommendations) is stored in module-level or class-level variables. When User A stores their last search and User B immediately overwrites it, User A's `save_recipe` call saves the wrong thing. This is a race condition — the hardest class of bug to reproduce and debug.

In this project: every WebSocket connection gets its own `SessionContext` object with its own scratchpad. They are completely isolated. There is no shared mutable state between users.

---

**Request C: "I want to write a test for the recipe-saving logic."**

In the monolith, `save_recipe()` calls the database directly. To test it, you need a real (or test) database running, the real LLM running, and often the whole server started. The test is slow, fragile, and depends on the environment.

In this project, `SaveRecipeTool` calls `RecipeManagerService`, which calls `RecipeRepository` — an *interface*, not a real class. In the test you hand it a fake repository that pretends to save. The test runs in milliseconds with no server, no database, no LLM.

---

**Request D: "We need to add a calorie-budget tool to the agent."**

In the monolith, you need to find the right place in 1 500 lines of `agent.py`, add the logic, update the inline prompt string somewhere in the middle of the file, and hope the surrounding code still works.

In this project: create one new file in `agent/tools/`, add one line in `factory.py`. The system prompt updates itself automatically based on which tools are registered.

---

### 2.3 Hexagonal Architecture: the core idea in one sentence

> **The business logic (domain) never imports or knows about the tools it uses. Instead, it publishes a list of contracts — "I need something that can save a user" — and the infrastructure fulfils those contracts.**

That's it. Everything else follows from that single principle.

---

### 2.4 A real-world analogy: the USB standard

Think about how USB works on your laptop.

Your laptop has USB ports. It does not care — and does not know — whether you plug in a keyboard, a mouse, a hard drive, a fan, or a phone charger. The laptop just exposes a *standard port* (a contract: "I will supply power and data in this format"). Any device that follows that contract works.

Now imagine the alternative: your laptop had a "keyboard-only port" soldered directly to one specific keyboard model from 2019. If that keyboard breaks, you need a new laptop. If a better keyboard comes out, you cannot use it.

Hexagonal Architecture applies this USB logic to software:

```
                    ┌──────────────────────────────────────┐
                    │                                      │
                    │         The Application Core         │
                    │     (domain/ + application/)         │
                    │                                      │
                    │   Defines PORTS (USB standards):     │
                    │   • "I need something that saves users"
                    │   • "I need something that parses intent"
                    │   • "I need something that finds recipes"
                    │                                      │
                    └──────────┬───────┬───────────────────┘
                               │       │
              ┌────────────────┘       └────────────────────┐
              │   ADAPTERS plug in here (USB devices)       │
              │                                             │
   ┌──────────▼──────────┐                    ┌────────────▼────────────┐
   │  LEFT SIDE          │                    │  RIGHT SIDE             │
   │  "Who calls us"     │                    │  "What we call"         │
   │                     │                    │                         │
   │  • Flutter app      │                    │  • SQLite database      │
   │  • REST API         │                    │  • Ollama / Groq / GPT  │
   │  • WebSocket chat   │                    │  • FAISS vectorstore    │
   │  • Tests            │                    │  • YOLO detector        │
   └─────────────────────┘                    └─────────────────────────┘
```

The core does not import `sqlite3`, `langchain`, or `faiss`. It only imports from `domain/ports.py` — its own published contracts. The implementations (SQLite, Ollama, FAISS) live entirely in `infrastructure/` and are plugged in by `factory.py` at startup.

---

### 2.5 What are "Ports" and "Adapters"? (with concrete examples)

**Ports** are Python `Protocol` classes in `domain/ports.py`. They describe a *capability* the application needs, with no implementation:

```python
# domain/ports.py — this is a PORT
# "I need something that can save a user and look one up by ID."
# Notice: no sqlite3, no SQL, no database anywhere.

class UserRepository(Protocol):
    async def get_by_id(self, user_id: int) -> User | None: ...
    async def save(self, user: User) -> int: ...
    async def soft_delete(self, user_id: int) -> None: ...
```

This is the USB socket on the laptop. The application says: "give me something that satisfies `UserRepository` and I will work with it."

**Adapters** are the concrete implementations in `infrastructure/`. They are the USB devices:

```python
# infrastructure/persistence/user_repo.py — this is an ADAPTER
# "Here is how you save a user specifically in SQLite."
# Notice: all the SQL lives here, isolated.

class SQLiteUserRepository:
    async def get_by_id(self, user_id: int) -> User | None:
        async with self._conn.acquire() as conn:
            row = await conn.execute_fetchall(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            )
            return self._row_to_user(row[0]) if row else None

    async def save(self, user: User) -> int:
        async with self._conn.acquire() as conn:
            cursor = await conn.execute(
                "INSERT INTO users (name, surname, ...) VALUES (?, ?, ...)",
                (user.name, user.surname, ...)
            )
            return cursor.lastrowid
```

To switch to PostgreSQL, you write `PostgresUserRepository` with the same three methods — and change one line in `factory.py`. The application layer never changes.

Here is every Port/Adapter pair in this project:

| Port (in `domain/ports.py`) | Adapter (in `infrastructure/`) | What it does |
|---|---|---|
| `UserRepository` | `SQLiteUserRepository` | Save/load users |
| `AuthenticationRepository` | `SQLiteAuthenticationRepository` | Login credentials |
| `MedicalRepository` | `SQLiteMedicalRepository` | Medical advice records |
| `RecipeRepository` | `SQLiteRecipeRepository` | Saved recipe history |
| `NutritionRepository` | `SQLiteNutritionRepository` | Calorie & nutrition logs |
| `ConversationRepository` | `SQLiteConversationRepository` | Chat session metadata |
| `ChatMessageRepository` | `SQLiteChatMessageRepository` | Individual chat messages |
| `IntentParserPort` | `IntentParser` (alias: `OllamaIntentParser`) | Parse user query → structured intent |
| `MedicalRAGPort` | `MedicalRAG` | Extract dietary rules from PDFs |
| `RecipeRAGPort` | `RecipeNutritionRAG` | Find recipes from dataset |
| `SafetyFilterPort` | `SafetyFilter` (alias: `OllamaSafetyFilter`) | Validate recipes against constraints |
| `IngredientDetectorPort` | `YOLOServiceDetector` / `LLaVAIngredientDetector` / `FallbackIngredientDetector` | Detect food in photos |

Every row is a USB socket + USB device pair. Swap the right column without touching the left.

---

### 2.6 The dependency rule: arrows always point inward

The single most important rule in Hexagonal Architecture:

> **Every import arrow must point toward the center. Outer layers may import inner layers. Inner layers must never import outer layers.**

```
        Outermost                              Innermost
        ┌──────────┐   imports   ┌───────────┐   imports   ┌────────┐
        │ adapters │ ──────────► │application│ ──────────► │ domain │
        └──────────┘             └───────────┘             └────────┘
                                                                ▲
        ┌────────────────┐                                      │
        │ infrastructure │ ─────────────────────────────────────┘
        └────────────────┘
           (also imports domain — to fulfil its ports)
```

What this means in practice:

- `domain/` imports **nothing** from the project. Only Python standard library.
- `application/` imports only from `domain/`.
- `infrastructure/` imports from `domain/` (to implement its ports) and external libraries (SQLite, LangChain, FAISS).
- `adapters/` imports from `application/` and `factory/`.
- `agent/` imports from `application/` (tools call services, not infrastructure directly).

If you ever open a file in `domain/` and see an import of `sqlite3`, `langchain`, or `fastapi` — that is a bug in the architecture. The domain must stay pure.

---

### 2.7 Side-by-side: monolith vs hexagonal for this exact project

The table below uses only situations that actually happened or would happen in this project.

| Scenario | Monolith | Hexagonal (this project) |
|---|---|---|
| **Switch LLM from Ollama to Groq** | Grep for `OllamaLLM` across 10+ files, rewrite each call site carefully | Set `AGENT_LLM_PROVIDER=groq` in `.env`. Change 0 lines of application logic. |
| **Add a second database (e.g. PostgreSQL for analytics)** | Refactor all `sqlite3.connect()` calls scattered across the codebase | Write `PostgresAnalyticsRepository` implementing `AnalyticsPort`. Register in `factory.py`. |
| **Two users chat simultaneously** | Shared global `agent_state` dict causes User B to read User A's last recipe search | Each connection gets its own `SessionContext` and `AgentExecutor`. Zero shared mutable state. |
| **LLM response takes 4 seconds** | Server blocks: nobody else gets served during those 4 seconds | Async: the server handles hundreds of other users while waiting for the LLM response |
| **Add a new agent tool (e.g. "show weekly nutrition summary")** | Find the right place in a 1500-line file, update inline prompts, test by running everything | Create `agent/tools/weekly_summary.py` (1 class, ~40 lines). Register in `factory.py`. Done. |
| **Test the recommendation pipeline** | Must have Ollama running, FAISS loaded, SQLite initialized, server started | Pass mock objects for every port. Test runs in 50ms with no external services. |
| **Change the recipe RAG prompt** | Search through a large file for the right string literal | Open `infrastructure/rag/recipe_rag.py`. Edit `SYSTEM_PROMPT`. One class, one attribute. |
| **User A's session crashes mid-request** | Can corrupt shared state for User B | Exception is caught and logged. User A gets a friendly error. User B is unaffected. |
| **Add a CLI interface alongside the REST API** | Clone and modify `agent.py` — two codebases diverge immediately | Create `adapters/cli/` that calls the same `ServiceFactory`. Zero duplication of business logic. |
| **Medical RAG returns bad results** | Debugging requires tracing through hundreds of lines of mixed concerns | `MedicalRAG` is one isolated class. Set a breakpoint, inspect inputs/outputs in isolation. |

---

### 2.8 Why the word "Hexagonal"?

The name comes from Alistair Cockburn's 2005 paper where he drew the application as a hexagon surrounded by adapters:

```
              [Test adapter]    [CLI adapter]
                    \               /
                     \             /
       [REST adapter]──────────────────[WebSocket adapter]
                    |                |
                    |   Application  |
                    |     Core       |
                    |                |
       [SQLite adapter]────────────────[Ollama adapter]
                     /             \
                    /               \
              [YOLO adapter]    [FAISS adapter]
```

The shape is a hexagon simply because it has enough sides to fit several adapters on each side visually — not because there must be exactly six. In practice you can have any number of adapters. What matters is the idea: the core in the center, the adapters all around it, each one plugging in through a port.

The alternative names — **Ports and Adapters** or **Clean Architecture** (Robert Martin's variant) — describe the same concept from different angles. In all of them the rule is identical: the center does not know about the outside.

---

### 2.9 Summary: the three things to remember

1. **Ports** are contracts (interfaces). They live in `domain/ports.py`. They say *what* is needed.
2. **Adapters** are implementations. They live in `infrastructure/`. They say *how* it is done.
3. **The Factory** (`factory.py`) is the only place that connects ports to adapters. Everything else stays ignorant of which adapter is currently plugged in.

If you remember only these three things, you will always know where to look when something needs to change.

---

## 3. Folder Structure Explained

```
src/
├── domain/           ← Layer 1: Pure business concepts. No imports from other src/ layers.
│   ├── entities.py   ← Persistent objects (User, Recipe, ChatMessage…)
│   ├── models.py     ← Value objects for the AI pipeline (UserIntent, Recipe, SafetyCheckResult…)
│   ├── ports.py      ← Interfaces (Protocols) — defines WHAT is needed, not HOW
│   └── exceptions.py ← Custom exceptions for every failure mode
│
├── application/      ← Layer 2: Orchestrates domain objects using ports
│   ├── context.py    ← SessionContext — per-user/per-request state, thread-safe
│   ├── dto.py        ← Data Transfer Objects: what services return to callers
│   └── services/     ← One service = one use case
│       ├── recommendation.py  ← The main 5-step AI pipeline
│       ├── recipe_manager.py  ← Save/list recipe history
│       ├── profile.py         ← User profile management
│       ├── image_analysis.py  ← CNN detection → recommendations
│       ├── chat_history.py    ← Conversation persistence
│       └── authentication.py  ← Register, login, JWT
│
├── infrastructure/   ← Layer 3: Implements the ports. External systems live here.
│   ├── config.py     ← Settings dataclass loaded from environment variables
│   ├── persistence/  ← SQLite repository implementations
│   │   ├── connection.py    ← Async SQLite connection with auto-commit/rollback
│   │   ├── migrations.py    ← Schema creation (CREATE TABLE IF NOT EXISTS)
│   │   ├── user_repo.py     ← Implements UserRepository port
│   │   ├── auth_repo.py     ← Implements AuthenticationRepository port
│   │   ├── medical_repo.py  ← Implements MedicalRepository port
│   │   ├── recipe_repo.py   ← Implements RecipeRepository port
│   │   ├── nutrition_repo.py← Implements NutritionRepository port
│   │   ├── profile_repo.py  ← Implements ProfileRepository port
│   │   ├── conversation_repo.py ← Implements ConversationRepository port
│   │   ├── chat_message_repo.py ← Implements ChatMessageRepository port
│   │   └── analytics_repo.py    ← Read-only aggregate queries
│   ├── llm/          ← LLM-backed implementations
│   │   ├── llm_builder.py   ← Single build_llm() function — all provider logic lives here
│   │   ├── intent_parser.py ← Implements IntentParserPort (multi-provider via llm_builder)
│   │   └── safety_filter.py ← Implements SafetyFilterPort (rules + LLM, multi-provider)
│   ├── rag/          ← Retrieval-Augmented Generation systems
│   │   ├── base_rag.py       ← Template method: vectorstore + chain setup
│   │   ├── medical_rag.py    ← Implements MedicalRAGPort — reads medical PDFs
│   │   ├── recipe_rag.py     ← Implements RecipeRAGPort — reads recipe CSVs
│   │   └── smart_retriever.py← Routes queries to recipe vs nutrition vectorstore
│   └── cnn/          ← Computer vision implementations
│       ├── ingredient_detector.py  ← LLaVA (multimodal LLM) detector
│       ├── yolo_service_detector.py← YOLO external service detector
│       └── fallback_detector.py    ← Try YOLO, fall back to LLaVA
│
├── agent/            ← Layer 4: The conversational AI brain
│   ├── executor.py   ← Runs the LLM + tool-selection loop (LangChain agent)
│   ├── memory.py     ← Per-session conversation history (not global)
│   ├── prompt.py     ← System prompt builder (dynamically includes registered tools)
│   └── tools/        ← Each tool is one capability the agent can invoke
│       ├── base.py          ← BaseTool ABC + ToolResult dataclass
│       ├── registry.py      ← Registers tools, wraps them for LangChain
│       ├── search_recipes.py← Calls RecommendationService
│       ├── save_recipe.py   ← Calls RecipeManagerService
│       ├── show_recipe.py   ← Retrieves recipe details from ctx.scratch
│       ├── analyze_image.py ← Calls ImageAnalysisService
│       ├── nutrition_status.py← Queries today's nutrition from DB
│       ├── safety_guard.py  ← Blocks dangerous system requests
│       ├── crisis_support.py← Responds to mental health crisis signals
│       └── general_chat.py  ← Handles off-topic/greeting messages
│
├── adapters/         ← Layer 5: Entry points (how the outside world calls in)
│   └── rest/
│       ├── app.py           ← FastAPI app, registers routers, initializes factory
│       ├── dependencies.py  ← JWT validation, factory access, SessionContext builder
│       ├── schemas.py       ← Pydantic models for HTTP request/response bodies
│       └── routers/         ← One file per API group
│           ├── auth.py           ← POST /auth/register, /auth/login, /auth/refresh
│           ├── chat_ws.py        ← WebSocket /ws/chat + GET /chat/history
│           ├── recommendations.py← POST /recommendations
│           ├── conversations.py  ← GET /conversations
│           ├── profile.py        ← GET/PUT /profile
│           ├── images.py         ← POST /upload/image
│           └── analytics.py      ← GET /analytics
│
└── factory.py        ← Composition root: wires all dependencies together

test_functionality/   ← Functional test scripts (run manually, outside pytest)
    ├── test_agent.py                         ← Interactive CLI chat with the full agent
    ├── test_intent_parser.py                 ← Test intent parsing in isolation
    ├── test_rag_medical.py                   ← Test Medical RAG constraints extraction
    ├── test_rag_recipe.py                    ← Test Recipe RAG recommendations
    ├── test_safety_filter.py                 ← Test safety filter with dummy recipes
    ├── test_search_recipes_tool.py           ← Test SearchRecipesTool end-to-end
    ├── test_add_dummy_user.py                ← Create a test user in the database
    ├── test_recreate_vector_db.py            ← Rebuild both FAISS vectorstores from scratch
    ├── test_add_data_to_medical_vector_db.py ← Append a PDF to Medical RAG vectorstore
    ├── test_add_data_to_recipe_vector_db.py  ← Append a CSV to Recipe RAG vectorstore
    └── test_add_data_to_vector_db.py         ← Deprecated (use the two specialised scripts above)

app/                  ← Flutter mobile app
└── nutrition_ai_assistent/lib/
    ├── main.dart              ← App entry point
    ├── services/
    │   ├── api_service.dart   ← HTTP client (REST calls)
    │   ├── auth_service.dart  ← Login/register/token storage
    │   ├── chat_ws_service.dart← WebSocket client for /ws/chat
    │   └── storage_service.dart← Secure token persistence
    └── screens/               ← One screen per user-facing feature
```

---

## 4. Layer-by-Layer Walkthrough

### 4.1 Domain — The Heart of the System

**Files:** `src/domain/`

The domain is the most important layer. It contains pure Python with **zero external dependencies** (no LangChain, no SQLite, no FastAPI). It defines:

#### `entities.py` — Things that live in the database

These are Python dataclasses with `id`, `created_at`, `updated_at`, `deleted_at` fields. They represent rows in the database but contain **no SQL code**.

```python
@dataclass
class User:
    id: Optional[int] = None
    name: str = ""
    # ... no sqlite3, no ORM
```

Why no ORM? SQLAlchemy or Django ORM would add framework coupling. The hexagonal approach keeps entities as plain dataclasses — any persistence technology can store them.

#### `models.py` — Value objects for the AI pipeline

These are **frozen** (immutable) dataclasses — snapshots of data flowing through the AI pipeline:

- `UserIntent` — what the LLM extracted from the user's message
- `NutritionConstraints` — dietary limits from medical PDFs
- `Recipe` — one recipe with ingredients, nutrition, instructions
- `SafetyCheckResult` — which recipes passed/failed safety validation
- `DetectedIngredients` — what the CNN saw in a food photo

They are frozen because the pipeline is a one-way flow: once intent is parsed, it doesn't change. Immutability prevents bugs where a downstream step accidentally mutates upstream data.

#### `ports.py` — The Contracts (Interfaces)

This is the key to swappability. Ports use Python's `typing.Protocol` — structural typing without inheritance:

```python
@runtime_checkable
class IntentParserPort(Protocol):
    async def parse(self, query: str) -> UserIntent: ...

@runtime_checkable
class UserRepository(Protocol):
    async def get_by_id(self, user_id: int) -> User | None: ...
    async def save(self, user: User) -> int: ...
    # ...
```

Any class that implements these methods satisfies the port — no `implements IntentParserPort` required. This means you can write a mock for tests without any special setup.

#### `exceptions.py` — Typed Errors

```
DomainError (base)
├── IntentParsingError   ← LLM couldn't understand the query
├── RAGError             ← RAG system failed
├── SafetyCheckError     ← Safety filter crashed
├── IngredientDetectionError ← CNN failed
├── RepositoryError      ← Database operation failed
├── AuthenticationError  ← Bad credentials / expired token
└── DuplicateLoginError  ← Login already taken
```

Typed exceptions let adapters (FastAPI routers) catch specific errors and return the right HTTP status code (e.g., `DuplicateLoginError` → 409 Conflict).

---

### 4.2 Application — Use Cases & Orchestration

**Files:** `src/application/`

The application layer contains the **use cases** — the things users actually want to do. Each service class orchestrates domain objects and ports. Services **never** directly instantiate repositories or LLM clients; they receive them via constructor injection.

#### `context.py` — SessionContext

```python
@dataclass
class SessionContext:
    user_id: int
    conversation_id: str
    user_data: dict        # health conditions, restrictions, preferences
    request_id: str        # unique per request, for tracing
    scratch: dict          # inter-tool scratchpad (not cleared between turns!)
```

`SessionContext` replaces the dangerous global `AgentState` from a monolith. Every function that needs user identity or scratchpad data receives a `SessionContext` explicitly. Two concurrent users → two completely separate `SessionContext` instances.

The `scratch` dict is especially important: when `SearchRecipesTool` runs, it stores the typed `RecommendationResult` in `ctx.scratch["last_recommendations"]`. When `SaveRecipeTool` runs a moment later, it reads from the same scratch to know which recipes were suggested — without the LLM having to re-transmit the entire recipe data.

#### `services/recommendation.py` — The Core Pipeline

This is the most important service. It runs a 5-step AI pipeline:

```
User query
    │
    ▼ Step 1: Intent Parsing
    │   OllamaIntentParser → UserIntent
    │   ("I have diabetes and want chicken" → {health_conditions: ["diabetes"], instructions: ["chicken"]})
    │
    ▼ Step 2: Medical Constraints
    │   Check DB cache first → else MedicalRAG → NutritionConstraints
    │   ({avoid: ["sugar"], constraints: {sugar_g: {max: 25}}})
    │
    ▼ Step 2.5: Daily Budget Adjustment
    │   Load today's saved meals from DB → subtract consumed from limits
    │   (If user already ate 15g sugar, max becomes 10g for new recipes)
    │
    ▼ Step 3: Query Augmentation
    │   Combine query + intent + constraints → enriched query string
    │
    ▼ Step 4: Recipe Retrieval
    │   RecipeNutritionRAG.async_ask() → list[Recipe]
    │
    ▼ Step 5: Safety Check
    │   OllamaSafetyFilter.check() → SafetyCheckResult
    │   (Filter UNSAFE recipes, flag warnings, format markdown)
    │
    ▼ RecommendationResult
```

Every step uses a port, never a concrete class. This is why you can swap LLM providers in `factory.py` without changing this file at all.

#### `services/authentication.py` — Auth

Handles:
- Password hashing with bcrypt
- JWT creation (`user_id` + `role` + expiry in payload)
- Token verification and refresh (for Flutter's token renewal without re-login)

---

### 4.3 Infrastructure — The Outside World

**Files:** `src/infrastructure/`

Infrastructure implements the ports defined in `domain/ports.py`. Each implementation knows about one external system (SQLite, Ollama, FAISS, etc.) and nothing else.

#### `persistence/connection.py` — Async SQLite

```python
class AsyncSQLiteConnection:
    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiosqlite.Connection]:
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                await conn.commit()   # auto-commit on success
            except Exception:
                await conn.rollback() # auto-rollback on failure
                raise
```

One shared `AsyncSQLiteConnection` object is created at startup (in `factory.py`) and passed to all repositories. Each repository call acquires a fresh connection context — there is no shared mutable connection state.

#### `persistence/migrations.py` — Schema

All `CREATE TABLE IF NOT EXISTS` statements live here. They run once at startup via `await run_migrations(connection)`. This replaces the scattered `create_all_tables()` calls in a monolith.

#### `persistence/*_repo.py` — Repository Implementations

Each repo implements one Repository Port. Example pattern:

```python
class SQLiteUserRepository:
    def __init__(self, connection: AsyncSQLiteConnection):
        self._conn = connection

    async def get_by_id(self, user_id: int) -> Optional[User]:
        async with self._conn.acquire() as conn:
            row = await conn.execute_fetchall(
                "SELECT * FROM users WHERE id = ? AND deleted_at = ''",
                (user_id,),
            )
            return self._row_to_user(row[0]) if row else None
```

Note: **soft deletes** — records are never physically deleted. `deleted_at` is set to a timestamp. This preserves audit history and allows data recovery.

#### `rag/base_rag.py` — RAG Template

The `BaseRAG` class implements the Template Method pattern:

```
initialize()
    ├── load embeddings (HuggingFace sentence-transformers)
    ├── build LLM (Ollama / Groq / OpenAI via _build_llm())
    ├── load or build FAISS vectorstore
    ├── _setup_retriever()  ← subclass may override
    └── _build_chain()      ← prompt + retriever + LLM chain
```

Subclasses only need to implement:
- `_ingest_documents()` — load raw data files
- `_ingest_single_file()` — load one file incrementally

Everything else (vectorstore management, chain building, LLM provider selection) is inherited.

#### `rag/medical_rag.py` — Medical PDFs

Reads PDF files from `data_test/raw/` using `PyPDFLoader`. Chunks them into 300-character pieces with 50-character overlap. When queried with a health condition like "diabetes", it retrieves relevant medical text and asks the LLM to return **structured JSON** with nutrition constraints:

```json
{
  "avoid": ["refined sugar", "white bread"],
  "constraints": {"sugar_g": {"max": 25}, "sodium_mg": {"max": 1500}},
  "notes": "Emphasize complex carbohydrates and fiber"
}
```

The result is parsed into a typed `NutritionConstraints` object and cached in the `medical_advice` table for future requests.

#### `rag/recipe_rag.py` — Recipe Datasets

Uses **dual vectorstores**:
- `recipes_and_meals_db` — recipe documents (ingredients, instructions, cuisine)
- `nutrition_facts_db` — per-ingredient nutrition data

The `SmartRetriever` analyzes the query keywords to decide which vectorstore(s) to search:
- "What recipe can I make?" → search `recipes`
- "How much protein in chicken?" → search `nutrition`
- "Healthy chicken dinner" → search both

The LLM is forced to return **JSON** with 3 structured recipes. These are parsed directly into `Recipe` domain objects — no second LLM call needed.

#### `llm/llm_builder.py` — Centralized LLM Factory

Before this file existed, every component that needed an LLM (`IntentParser`, `SafetyFilter`, `BaseRAG`) contained its own `if provider == "groq": ... elif provider == "openai": ... else: ...` block. That was duplicated code — if a new provider appeared, or a constructor argument changed, every component had to be updated independently.

`llm_builder.py` extracts all of that logic into a single `build_llm()` function:

```python
def build_llm(
    *,
    provider: str,         # "openai" | "groq" | "ollama"
    model: str,
    temperature: float = 0,
    json_mode: bool = False,      # forces JSON output format
    chat_model: bool = False,     # ChatOllama vs OllamaLLM (for tool-calling)
    max_tokens: Optional[int] = None,
    ollama_base_url: str = "http://localhost:11434/",
    openai_api_key: str = "",
    groq_api_key: str = "",
) -> Union[BaseChatModel, BaseLLM]:
```

Now every component that needs an LLM calls `build_llm(provider=..., model=..., ...)` and the factory function returns the right LangChain object. To add a fourth provider (e.g. Anthropic), you add one branch in `llm_builder.py` — and every component automatically gains support for it.

Note the `chat_model` flag for Ollama: `ChatOllama` supports tool-calling (needed by the agent executor), while `OllamaLLM` is a plain text completion model (used by intent parser and safety filter for JSON generation).

#### `llm/intent_parser.py` — Intent Parser (multi-provider)

Implements `IntentParserPort`. Receives a raw user query string and returns a typed `UserIntent` object with structured fields (`name`, `health_conditions`, `restrictions`, `preferences`, `instructions`).

Now delegates LLM construction to `build_llm()`. The backward-compatible alias `OllamaIntentParser = IntentParser` keeps existing imports working while the class itself supports all three providers.

#### `llm/safety_filter.py` — Hybrid Safety Check

Two-phase validation for each recipe:

**Phase 1 — Rule-based (fast, deterministic):**
- Checks ingredient list against `avoid_foods` from medical profile
- Checks restrictions (vegetarian, gluten-free…) against ingredient keyword map
- Checks nutrition values against numeric limits

**Phase 2 — LLM semantic check (catches subtle violations):**
- Asks LLM: "prosciutto is pork — flag if pescatarian"
- "soy sauce contains gluten — flag if gluten-free"
- Returns SAFE / WARNING / UNSAFE verdict per recipe

#### `cnn/` — Image Detection

Three implementations of `IngredientDetectorPort`:

| Class | Technology | When used |
|---|---|---|
| `LLaVAIngredientDetector` | LLaVA vision model via Ollama | Always available, slower |
| `YOLOServiceDetector` | YOLO in separate Docker container | Fast, but container must be running |
| `FallbackIngredientDetector` | Try YOLO, fall back to LLaVA | Default (best of both) |

---

### 4.4 Agent — The Conversational Brain

**Files:** `src/agent/`

The agent is the conversational interface. It wraps LangChain's `create_tool_calling_agent` with custom tools and per-session memory.

#### `executor.py` — AgentExecutor

```
AgentExecutor.run(ctx, user_input)
    │
    ├── First call only:
    │   ├── Load conversation history from DB → in-memory list
    │   └── Build LangChain executor (prompt + tools + LLM)
    │
    ├── Store user_input in ctx.scratch["original_query"]  ← prevents LLM from rewriting it
    │
    ├── executor.invoke({input, chat_history})
    │   └── LLM selects tool → tool executes → LLM formats response
    │
    ├── For certain tools (search_recipes, analyze_image, nutrition_status):
    │   └── Return tool's raw output verbatim (prevent LLM from truncating recipe markdown)
    │
    ├── Update in-memory history
    └── Persist both messages to DB (non-blocking)
```

Why store `original_query` in scratch? The LLM agent sometimes *rewrites* the user's query before passing it to tools. For `search_recipes`, we always want the original verbatim message to reach the intent parser — because the intent parser extracts structured data from natural language, and any LLM rewriting could lose information.

#### `memory.py` — ConversationMemory

```python
class ConversationMemory:
    def __init__(self, max_messages=50, chat_history_service=None):
        self._messages: list[BaseMessage] = []  # NOT global
```

Per-session, per-user. The `max_messages=50` cap prevents the LLM context window from overflowing in long conversations. On WebSocket reconnect, history is loaded from the database so the user can continue where they left off.

#### `prompt.py` — System Prompt Builder

The system prompt is **generated dynamically** based on which tools are registered. If you remove the `crisis_support` tool, its routing rule and example disappear from the prompt automatically. This prevents stale prompt instructions for non-existent tools.

```python
def build_system_prompt(registry: ToolRegistry) -> str:
    has_crisis_support = "crisis_support" in registry.names()
    crisis_rule = "\n5. When user expresses suicidal thoughts..." if has_crisis_support else ""
    return f"""You are a helpful nutrition assistant...
    {crisis_rule}
    ..."""
```

#### `tools/base.py` — Tool Interface

```python
class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    async def execute(self, ctx: SessionContext, **kwargs) -> ToolResult: ...

    @abstractmethod
    def get_schema(self) -> type[BaseModel]: ...
```

Every tool:
- Has a `name` the LLM uses to call it
- Has a `description` the LLM reads to decide when to use it
- Takes a `SessionContext` (for user identity, scratchpad)
- Returns a `ToolResult` with `output` (shown to LLM/user) and optionally `data` + `store_as` (stored silently in `ctx.scratch`)

#### `tools/registry.py` — Tool Registry

The registry:
1. Stores all tools by name
2. Converts them to LangChain `StructuredTool` objects (needed for `create_tool_calling_agent`)
3. Binds `SessionContext` to each tool via closure (so LangChain can call them without knowing about sessions)

The LangChain executor runs tool calls synchronously inside a thread pool. The registry wrapper calls `asyncio.run(tool.execute(...))` to bridge back to async — this works because the thread pool thread has no running event loop.

---

### 4.5 Adapters — Entry Points

**Files:** `src/adapters/rest/`

Adapters translate between the outside world (HTTP, WebSocket, JSON) and the application layer (services, DTOs).

#### `app.py` — FastAPI Application

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Settings.from_env()
    factory = ServiceFactory(config)
    await factory.initialize()   # runs DB migrations + initializes RAG
    set_factory(factory)         # stores globally for dependency injection
    yield
```

The `lifespan` context manager runs exactly once at startup. This is where the expensive operations happen: RAG vectorstores are loaded (or built), LLM clients are created, DB migrations run.

#### `dependencies.py` — Shared FastAPI Dependencies

```python
async def get_current_user(credentials, factory) -> CurrentUser:
    payload = auth_service.verify_token(credentials.credentials)
    return CurrentUser(user_id=payload["user_id"], role=payload["role"])

async def build_session_ctx(user_id, conversation_id, factory) -> SessionContext:
    user_data = await profile_svc.load_user_context(user_id)  # loads health profile
    return SessionContext(user_id=user_id, conversation_id=conversation_id, user_data=user_data)
```

`build_session_ctx` pre-populates the session with the user's saved health profile. This means every subsequent recommendation automatically respects conditions set during registration — the user doesn't need to repeat them every message.

#### `routers/chat_ws.py` — WebSocket Chat

```
Client connects: ws://server/ws/chat?token=<JWT>
    │
    ├── Validate JWT token
    ├── Resolve conversation_id (reuse last active, or new)
    ├── Cleanup old messages (soft-delete > 48h)
    ├── Build SessionContext (loads user health profile from DB)
    ├── Create AgentExecutor (factory.create_agent(ctx))
    │
    └── Loop:
        ├── receive_text() → agent.run(ctx, message) → send_text(response)
        └── repeat until disconnect
```

Conversation reuse: if the user's last conversation was active within 48 hours, the same `conversation_id` is reused. This means the Flutter app reconnecting next morning will see the previous conversation context.

#### `schemas.py` — API Schemas

Pydantic models that validate HTTP request/response bodies. Completely separate from domain entities — this is intentional. API schemas can add validation rules, rename fields for the API, or add/remove fields without touching domain entities.

---

### 4.6 Factory — The Composition Root

**File:** `src/factory.py`

The factory is where all the wiring happens. It is the **only** place in the codebase that knows about concrete implementations. Everything else works with interfaces (ports).

```python
class ServiceFactory:
    async def initialize(self):
        # Run DB migrations
        await run_migrations(self._connection)

        # Build shared expensive singletons ONCE
        self._intent_parser = OllamaIntentParser(...)
        self._safety_filter = OllamaSafetyFilter(...)
        self._agent_llm = self._build_agent_llm()      # Groq / OpenAI / Ollama
        self._image_detector = self._build_image_detector()
        self._medical_rag = MedicalRAG(...)
        self._recipe_rag = RecipeNutritionRAG(...)

    def create_recommendation_service(self) -> RecommendationService:
        # Inject concrete implementations into service expecting ports
        return RecommendationService(
            intent_parser=self._intent_parser,    # satisfies IntentParserPort
            medical_rag=self._medical_rag,        # satisfies MedicalRAGPort
            recipe_rag=self._recipe_rag,          # satisfies RecipeRAGPort
            safety_filter=self._safety_filter,    # satisfies SafetyFilterPort
            medical_repo=SQLiteMedicalRepository(self._connection),  # satisfies MedicalRepository
            nutrition_repo=SQLiteNutritionRepository(self._connection),
        )
```

The LLM clients (`intent_parser`, `safety_filter`, `agent_llm`, RAGs) are expensive to create — they load models, establish connections. They are built **once** and reused across all requests.

Repositories are cheap (just a reference to a connection). They are created fresh per-request — this is safe because they hold no mutable state.

---

## 5. Complete Request Flow

### 5.1 Chat Message (WebSocket)

```
Flutter app sends: "I have diabetes, find me a low-sugar dinner"
    │
    │ [WebSocket frame]
    ▼
chat_ws.py: router receives text
    │ agent.run(ctx, "I have diabetes, find me a low-sugar dinner")
    ▼
executor.py: AgentExecutor.run()
    │ stores message in ctx.scratch["original_query"]
    │ invokes LangChain executor with chat_history
    ▼
LangChain agent (llama3.2):
    │ reads system prompt + routing rules
    │ decides: call "search_recipes"
    ▼
agent/tools/search_recipes.py: SearchRecipesTool.execute()
    │ reads ctx.scratch["original_query"] (not LLM-rewritten query!)
    ▼
application/services/recommendation.py: RecommendationService.get_recommendations()
    │
    ├── Step 1: intent_parser.parse(query) → UserIntent
    │           {health_conditions: ["diabetes"]}
    │
    ├── Step 2: medical_rag.get_constraints(["diabetes"])
    │           or DB cache → NutritionConstraints
    │           {avoid: ["sugar"], constraints: {sugar_g: {max: 25}}}
    │
    ├── Step 2.5: nutrition_repo.get_today_by_user(user_id)
    │             → adjust constraints for today's consumed food
    │
    ├── Step 3: _build_augmented_query() → enriched query string
    │
    ├── Step 4: recipe_rag.async_ask(augmented_query) → list[Recipe]
    │
    └── Step 5: safety_filter.check(recipes, constraints, intent)
                → SafetyCheckResult (filtered markdown + verdicts)
    │
    ▼ RecommendationResult stored in ctx.scratch["last_recommendations"]
    │
    ▼ search_recipes returns markdown: "## 1. Grilled Salmon\n..."
    │
executor.py: detects search_recipes in _DIRECT_OUTPUT_TOOLS → returns raw markdown verbatim
    │
    │ persists user message + agent response to DB
    ▼
Flutter app receives: formatted recipe recommendations
```

### 5.2 Recipe Recommendation Pipeline

See the 5-step pipeline in [Section 4.2](#42-application--use-cases--orchestration) above.

### 5.3 Image Analysis Flow

```
User sends: "what can I cook? [IMAGE:/uploads/fridge.jpg]"
    │
agent routes to: analyze_image tool
    │
AnalyzeImageTool.execute(ctx, image_path="/uploads/fridge.jpg")
    │
ImageAnalysisService.analyze(ctx, image_path)
    │
    ├── detector.detect("/uploads/fridge.jpg")
    │   ├── YOLOServiceDetector (http://yolo-detector:8001)
    │   │   → POST /detect with image → ["chicken", "garlic", "lemon"]
    │   └── On failure: LLaVAIngredientDetector
    │       → base64 encode → Ollama LLaVA → parse ingredient list
    │
    └── recommendation_service.get_recommendations(ctx, "ingredients: chicken, garlic, lemon")
        → full 5-step pipeline
    │
ImageAnalysisResult(detected=DetectedIngredients, recommendation=RecommendationResult)
```

---

## 6. The Flutter App

**Location:** `app/nutrition_ai_assistent/lib/`

The Flutter app is a thin client — it does not contain any business logic. All AI processing happens on the server.

### Services

| File | Role |
|---|---|
| `api_service.dart` | HTTP client with JWT auth headers. Base URL configured via `--dart-define=API_BASE_URL` |
| `auth_service.dart` | POST /auth/register, /auth/login, token refresh logic |
| `chat_ws_service.dart` | Connects to `ws://server/ws/chat?token=<JWT>`, streams messages |
| `storage_service.dart` | Persists JWT token securely on device |

### WebSocket Communication

```dart
// Connect
_channel = WebSocketChannel.connect(Uri.parse('$wsBase/ws/chat?token=$token'));
await _channel!.ready;

// Send a message
_channel!.sink.add("I need a healthy dinner");

// Receive agent response
_channel!.stream.listen((data) => onMessage(data.toString()));
```

The protocol is dead simple: send plain text, receive plain text (markdown from the agent). No JSON envelope needed for chat.

### Image Upload Flow

1. User selects photo → `XFile.readAsBytes()` (works on all platforms including web)
2. `api_service.uploadImageBytes()` → POST `/upload/image` → server returns `{"path": "/uploads/abc123.jpg"}`
3. App appends `[IMAGE:/uploads/abc123.jpg]` to the message text
4. Agent receives the message, extracts the path, calls `analyze_image` tool

---

## 7. Docker Deployment

Three containers work together:

```yaml
# docker-compose.yml
services:
  ollama:          # Local LLM inference (llama3.2, llava)
    port: 11434
  yolo-detector:   # YOLO + ResNet food detection microservice
    port: 8001
  api:             # FastAPI application (src/)
    port: 8000
    depends_on: [ollama, yolo-detector]
```

The API container calls the other two by their container names (`ollama`, `yolo-detector`) which Docker's internal DNS resolves. The Flutter app talks only to the API container on port 8000.

---

## 8. How to: Add a New Tool

Adding a new tool to the agent is a 3-step process. No other files need modification.

### Step 1 — Create the tool file

Create `src/agent/tools/my_new_tool.py`:

```python
from pydantic import BaseModel, Field
from application.context import SessionContext
from agent.tools.base import BaseTool, ToolResult


class MyNewToolInput(BaseModel):
    """Pydantic schema — the LLM uses field descriptions to fill in arguments."""
    some_param: str = Field(description="What this parameter means to the LLM")


class MyNewTool(BaseTool):
    name = "my_new_tool"
    description = (
        "Describe WHEN the agent should call this tool. "
        "Be specific — the LLM reads this to decide whether to use it."
    )

    def __init__(self, some_service):
        self._service = some_service  # inject dependencies

    def get_schema(self) -> type[BaseModel]:
        return MyNewToolInput

    async def execute(self, ctx: SessionContext, some_param: str = "", **kwargs) -> ToolResult:
        # ctx.user_id — who is asking
        # ctx.scratch — shared scratchpad (read last_recommendations, etc.)
        # ctx.user_data — health profile loaded from DB

        result = await self._service.do_something(ctx.user_id, some_param)

        return ToolResult(
            output=f"Here is the result: {result}",  # shown to LLM and user
            data=result,                               # optional structured data
            store_as="my_result_key",                  # optional: store in ctx.scratch
        )
```

### Step 2 — Register in `factory.py`

In `src/factory.py`, inside `create_agent()`:

```python
from agent.tools.my_new_tool import MyNewTool

def create_agent(self, ctx: SessionContext) -> AgentExecutor:
    # ... existing tools ...
    registry.register(MyNewTool(some_service=self._some_service))
    # Done! The system prompt updates automatically.
```

If your tool needs a service that does not exist yet, create it in `src/application/services/` and add a `create_X_service()` method to the factory following the same pattern as existing services.

### Step 3 — (Optional) Add a routing hint in `prompt.py`

The system prompt already says "Use tools based on their descriptions." For most tools, the description on the tool itself is sufficient. But if your tool needs a very specific routing rule (like `crisis_support` which must take absolute priority), add a rule in `src/agent/prompt.py`:

```python
has_my_tool = "my_new_tool" in tool_names
my_tool_rule = (
    "\n10. When user asks about X → call 'my_new_tool'."
) if has_my_tool else ""

return f"""...existing prompt...{my_tool_rule}"""
```

### Testing your tool

Because tools receive injected services, you can test them without a running LLM or database:

```python
# test_my_tool.py
from unittest.mock import AsyncMock
from agent.tools.my_new_tool import MyNewTool
from application.context import SessionContext

async def test_my_new_tool():
    mock_service = AsyncMock()
    mock_service.do_something.return_value = "test result"

    tool = MyNewTool(some_service=mock_service)
    ctx = SessionContext(user_id=1, conversation_id="test-conv")

    result = await tool.execute(ctx, some_param="hello")
    assert "test result" in result.output
```

---

## 9. How to: Change the Database

The database is SQLite, accessed through async repositories. The schema is in `src/infrastructure/persistence/migrations.py`.

### 9.1 Add a new column to an existing table

**Example:** Add `weight_kg` to the `users` table.

**Step 1 — Update the migration**

In `src/infrastructure/persistence/migrations.py`, add the column to the `CREATE TABLE` statement:

```python
"""CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    ...
    weight_kg REAL          -- ADD THIS
)"""
```

> **Note:** `CREATE TABLE IF NOT EXISTS` only runs when the table does not exist. For an existing database, you also need an `ALTER TABLE` migration. Add it as a separate step:

```python
_MIGRATIONS = [
    # Always runs (idempotent table creation)
    "CREATE TABLE IF NOT EXISTS users (...)",
    # One-time column addition — safe to run multiple times if you add IF NOT EXISTS guard
    "ALTER TABLE users ADD COLUMN weight_kg REAL DEFAULT NULL",
]
```

**Step 2 — Update the entity**

In `src/domain/entities.py`:

```python
@dataclass
class User:
    id: Optional[int] = None
    name: str = ""
    # ... existing fields ...
    weight_kg: Optional[float] = None   # ADD THIS
```

**Step 3 — Update the repository**

In `src/infrastructure/persistence/user_repo.py`, update the `save()` and `_row_to_user()` methods:

```python
async def save(self, user: User) -> int:
    async with self._conn.acquire() as conn:
        cursor = await conn.execute(
            """INSERT INTO users (name, ..., weight_kg, ...)
               VALUES (?, ..., ?, ...)""",
            (user.name, ..., user.weight_kg, ...),
        )
        return cursor.lastrowid

@staticmethod
def _row_to_user(row) -> User:
    return User(
        id=row[0], name=row[1], ...,
        weight_kg=row[10] or None,   # map the new column
    )
```

**Step 4 — Update the REST schema** (if exposed via API)

In `src/adapters/rest/schemas.py`:

```python
class UserOut(BaseModel):
    name: str
    # ...
    weight_kg: Optional[float] = None   # ADD THIS
```

**That's it.** No other files need changes.

### 9.2 Add a new table

**Step 1** — Add a `CREATE TABLE` statement to `migrations.py`.

**Step 2** — Add a corresponding entity dataclass to `domain/entities.py`.

**Step 3** — Add a Repository Port to `domain/ports.py`:

```python
@runtime_checkable
class MyNewRepository(Protocol):
    async def save(self, record: MyNewEntity) -> int: ...
    async def get_by_user(self, user_id: int) -> list[MyNewEntity]: ...
```

**Step 4** — Implement the repository in `infrastructure/persistence/my_new_repo.py`.

**Step 5** — Register in `factory.py`:

```python
def create_something_service(self) -> SomethingService:
    return SomethingService(
        my_new_repo=SQLiteMyNewRepository(self._connection),
    )
```

### 9.3 Switch from SQLite to PostgreSQL

Because all database access goes through Repository Ports, switching databases means:

1. Create `infrastructure/persistence/postgres/` with Postgres implementations of every port.
2. Change `factory.py` to instantiate `PostgresUserRepository` instead of `SQLiteUserRepository`.
3. Update `migrations.py` with Postgres-compatible DDL.

The domain, application, and agent layers are **completely unchanged**.

---

## 10. How to: Change the Agent Prompt

The agent prompt is in `src/agent/prompt.py`, in the `build_system_prompt()` function.

### What the system prompt controls

- **Tool routing rules:** which phrases trigger which tools ("when user asks for recipes → call search_recipes")
- **Priority order:** crisis_support always beats everything else
- **Output format guidance:** "always pass the verbatim message to search_recipes"
- **Workflow examples:** show the LLM exactly how to chain tool calls

### How to edit it

Open `src/agent/prompt.py` and find the `return f"""..."""` at the bottom of `build_system_prompt()`. Edit the text directly.

```python
return f"""You are a helpful nutrition assistant...

TOOL ROUTING RULES:
1. ...your changes here...
"""
```

**Important constraints:**
- Keep `{show_rule}`, `{nutrition_status_rule}`, etc. interpolation variables — they are populated from registered tools.
- Always include `crisis_support` rules with the highest priority.
- The LLM (especially smaller models like llama3.2) needs **concrete examples**, not just abstract rules. The `WORKFLOW EXAMPLES:` section is critical for correct tool selection.

### How to add a per-user health context section

The agent also receives a dynamic "USER HEALTH PROFILE" block injected by `executor.py`'s `_build_health_context()`. To add new profile fields to this context, edit `_build_health_context()` in `src/agent/executor.py`.

---

## 11. How to: Change the RAG Prompt

There are two RAG systems with separate prompts.

### Medical RAG Prompt

**File:** `src/infrastructure/rag/medical_rag.py`

```python
class MedicalRAG(BaseRAG):
    SYSTEM_PROMPT = """You are a medical nutrition specialist...

    You MUST respond with a valid JSON object using EXACTLY this structure:
    {
      "dietary_goals": "...",
      "avoid": ["..."],
      "constraints": { "sugar_g": {"max": <number>} },
      ...
    }
    """
```

**What it does:** Given medical PDF chunks as context, extracts structured dietary constraints as JSON.

**When to change it:**
- Add a new field to the JSON response (also update `MedicalRAG._parse_constraints_response()` and `NutritionConstraints` in `domain/models.py`)
- Change the output format
- Improve extraction accuracy for specific conditions

**Critical:** The `{input}` and `{context}` placeholders **must** remain in the prompt — they are required by LangChain's retrieval chain. The `BaseRAG._build_chain()` method will raise an error if they're missing.

### Recipe RAG Prompt

**File:** `src/infrastructure/rag/recipe_rag.py`

```python
class RecipeNutritionRAG(BaseRAG):
    SYSTEM_PROMPT = """You are NutriGuide, an AI nutrition assistant...

    OUTPUT: Return ONLY valid JSON, no extra text, no markdown fences.
    {
      "recipes": [
        {
          "name": "...",
          "ingredients": ["200g chicken", "..."],
          "nutrition": {"calories": 350, "protein_g": 25, ...}
        }
      ]
    }

    User Query: {input}
    Context: {context}
    """
```

**What it does:** Given recipe CSV rows as context, generates 3 structured recipe recommendations.

**When to change it:**
- Change the number of recipes returned (currently 3)
- Add new fields to the recipe output (also update `RecipeNutritionRAG._parse_json_to_recipes()` and `Recipe` in `domain/models.py`)
- Change prioritization rules (allergens, medical needs, preferences)

### Updating RAG prompts at runtime (advanced)

`BaseRAG` supports runtime prompt updates via:

```python
medical_rag.update_system_prompt(new_prompt_string)
# rebuilds the LangChain chain immediately
```

This allows changing the prompt without restarting the server, but use with caution — the new prompt must still contain `{input}` and `{context}`.

---

## 12. How to: Switch or Change the LLM Model

All LLM provider configuration flows through environment variables → `Settings` → `ServiceFactory`.

### Which LLMs are used where

| Component | Config key | Role |
|---|---|---|
| Intent Parser | `LLM_MODEL` + `LLM_PROVIDER` | Extract UserIntent JSON from user message |
| Safety Filter | `LLM_MODEL` + `LLM_PROVIDER` | Semantic recipe safety check |
| Medical RAG | `RAG_LLM_PROVIDER` + `RAG_LLM_MODEL` | Extract dietary constraints from PDFs |
| Recipe RAG | `RAG_LLM_PROVIDER` + `RAG_LLM_MODEL` | Generate recipe recommendations |
| Agent | `AGENT_LLM_PROVIDER` + `AGENT_LLM_MODEL` | Main conversational agent, tool routing |
| Image Detection | `CNN_DETECTOR_TYPE` + `cnn_model_path` | YOLO or LLaVA for food photos |

### Switch the Agent LLM

**To Groq (cloud, fast):**
```env
AGENT_LLM_PROVIDER=groq
AGENT_LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...
```

**To OpenAI:**
```env
AGENT_LLM_PROVIDER=openai
AGENT_LLM_MODEL_OPENAI=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

**To a different Ollama model:**
```env
AGENT_LLM_PROVIDER=ollama
AGENT_LLM_MODEL=mistral:7b
OLLAMA_BASE_URL=http://localhost:11434/
```

### Switch the RAG LLM

```env
RAG_LLM_PROVIDER=groq
RAG_LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...
```

> The Medical RAG and Recipe RAG both use `RAG_LLM_PROVIDER`/`RAG_LLM_MODEL`. They can't currently be set to different providers independently. To achieve that, add separate config keys and pass them in `factory.py`'s `initialize()`.

### How provider switching works in code

All provider-switching logic is now centralized in **`src/infrastructure/llm/llm_builder.py`** via the `build_llm()` function. Every component that needs an LLM (`IntentParser`, `SafetyFilter`, `BaseRAG`) calls this single function:

```python
# infrastructure/llm/llm_builder.py
def build_llm(*, provider, model, temperature=0, json_mode=False, ...):
    if provider == "openai":
        return ChatOpenAI(model=model, ...)
    elif provider == "groq":
        return ChatGroq(model=model, ...)
    elif provider == "ollama":
        return ChatOllama(model=model, ...)  # or OllamaLLM for non-chat
    else:
        raise ValueError(f"Unsupported provider: '{provider}'")
```

`ServiceFactory._build_agent_llm()` in `factory.py` also uses the same `build_llm()` call for the conversational agent LLM, keeping all provider logic in one file.

### Add support for a completely new LLM provider

1. Add the new API key and model name fields to `Settings` in `src/infrastructure/config.py`.
2. Add a new `elif provider == "your_provider":` branch in **`src/infrastructure/llm/llm_builder.py`**. That's the single place — `IntentParser`, `SafetyFilter`, all RAGs, and the agent executor all get the new provider automatically.
3. Install the corresponding LangChain integration package.
4. Set the environment variable in `.env`.

---

## 13. Configuration Reference

All configuration lives in environment variables, loaded by `src/infrastructure/config.py`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Ollama model for intent parser & safety filter |
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace model for FAISS vectorstores |
| `OLLAMA_BASE_URL` | `http://localhost:11434/` | Ollama API endpoint |
| `RAG_LLM_PROVIDER` | `ollama` | Provider for RAG systems: `ollama`, `groq`, `openai` |
| `RAG_LLM_MODEL` | `llama3.2` | Model for RAG systems |
| `AGENT_LLM_PROVIDER` | `ollama` | Provider for conversational agent |
| `AGENT_LLM_MODEL` | `llama3.2` | Ollama model for agent |
| `AGENT_LLM_MODEL_OPENAI` | `gpt-4.1-mini` | OpenAI model for agent (when provider=openai) |
| `AGENT_MAX_ITERATIONS` | `5` | Max tool-calling loop iterations per message |
| `GROQ_API_KEY` | — | Required when any provider is `groq` |
| `OPENAI_API_KEY` | — | Required when any provider is `openai` |
| `DB_PATH` | `users.db` | Path to SQLite database file |
| `CNN_DETECTOR_TYPE` | `yolo_with_fallback` | `llava_only`, `yolo_only`, `yolo_with_fallback` |
| `YOLO_SERVICE_URL` | `http://localhost:8001` | URL of the YOLO detector microservice |
| `JWT_SECRET` | `change-me-in-production` | **Change this in production!** |
| `JWT_EXPIRY_HOURS` | `24` | JWT token lifetime |

Create a `.env` file in the project root for local development. In Docker, these are set via `env_file: .env` in `docker-compose.yml`.

---

## 14. Functional Testing Scripts

**Location:** `test_functionality/`

These are standalone Python scripts for manually testing individual components of the system **outside** pytest. They are useful when:
- You want to interact with a live Ollama instance and see the raw output
- You need to rebuild or extend the FAISS vectorstores after adding new data files
- You want to debug one component in isolation (e.g. only the intent parser, only the safety filter)
- You need to seed the database with a test user before running other tests

All scripts follow the same pattern:
```python
# 1. Add src/ to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 2. Load config from environment (.env file)
settings = Settings.from_env()

# 3. Build and initialize the full service graph
factory = ServiceFactory(settings)
await factory.initialize()

# 4. Grab the component to test directly from the factory
component = factory._some_component

# 5. Run and print results
result = await component.do_something(...)
print(result)
```

### Script Reference

| Script | What it does | When to use |
|---|---|---|
| `test_agent.py` | Interactive CLI chat loop with the full agent (LLM + all tools) | End-to-end smoke test of the whole system |
| `test_intent_parser.py` | Parses a hardcoded query string and prints the `UserIntent` | Debug intent extraction; change the `text` variable |
| `test_rag_medical.py` | Queries Medical RAG for a list of conditions and prints `NutritionConstraints` | Debug medical constraints extraction |
| `test_rag_recipe.py` | Queries Recipe RAG with a free-text query and prints structured recipes | Debug recipe retrieval quality |
| `test_safety_filter.py` | Runs the safety filter against hardcoded dummy recipes and constraints | Debug rule-based + LLM safety checks |
| `test_search_recipes_tool.py` | Calls `SearchRecipesTool` end-to-end with a hardcoded query | Test the full recommendation pipeline via the agent tool interface |
| `test_add_dummy_user.py` | Creates a test user (`id=1`, `testuser`) in the SQLite DB | Required before running other tests that need a user in the DB |
| `test_recreate_vector_db.py` | Calls `initialize(force_rebuild=True)` on both RAGs to wipe and rebuild FAISS indexes | After changing embedding models or ingestion logic |
| `test_add_data_to_medical_vector_db.py` | Prompts for a PDF path and appends it to the Medical RAG vectorstore | Add new medical guidelines without full rebuild |
| `test_add_data_to_recipe_vector_db.py` | Prompts for a CSV path and appends it to the Recipe RAG vectorstore (with large-file batching for big CSVs) | Add new recipe datasets without full rebuild |
| `test_add_data_to_vector_db.py` | **Deprecated** — prints a message redirecting to the two scripts above | Don't use |

### How to run

```bash
# From the project root (requires .env to be present)
cd test_functionality
python test_agent.py

# Or from the project root directly
python test_functionality/test_agent.py
```

The scripts load `.env` via `Settings.from_env()`, so make sure your environment variables (Ollama URL, API keys, DB path, etc.) are set before running.

---

*This document was written for the `ievgen-app` branch as of February 2026. When making significant architectural changes, please update the relevant sections.*
