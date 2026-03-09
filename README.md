# Charly_agent

Agent backend + FastAPI API + React frontend.

## Architecture

- HTTP API: `apps/api`
- Web frontend: `apps/web`
- Agent core: `src/agent`

## Folders and files

### Root

- `apps/`: runnable applications (API and frontend).
- `src/`: Python code for the `agent` package.
- `tests/`: unit tests and API backend tests.
- `docs/`: functional and technical documentation.
- `workspace/`: local workspace used by some tools.
- `.env.example`: reference environment variables.
- `pyproject.toml`: Python package config and dependencies.
- `package.json`: root dev scripts (`dev`, `dev:api`, `dev:web`).
- `package-lock.json`: npm lockfile for the root project.
- `.gitignore`: Git exclusion rules.
- `README.md`: setup and usage guide.
- `TODO.md`: technical/product backlog.

### `apps/api` (FastAPI transport layer)

- `app.py`: builds the FastAPI application (CORS + routers).
- `main.py`: ASGI entrypoint (`create_app`) for `uvicorn`.
- `dependencies.py`: creates/injects `RuntimeContext`.
- `schemas.py`: Pydantic schemas for endpoints.
- `sse.py`: SSE event encoding.
- `routes/health.py`: health endpoint (`/api/health`).
- `routes/chat.py`: chat streaming endpoint (`/api/chat/stream`).

### `apps/web` (React/Vite)

- `index.html`: web app HTML template.
- `package.json`: frontend scripts/dependencies.
- `vite.config.ts`: Vite config + API proxy.
- `tsconfig.json`: TypeScript configuration.
- `src/main.tsx`: React entrypoint.
- `src/App.tsx`: chat/UI orchestration.
- `src/components/*`: UI components.
- `src/lib/api.ts`: HTTP client for the backend API.
- `src/lib/stream.ts`: SSE parsing/consumption.
- `src/types/chat.ts`: chat domain TypeScript types.
- `src/styles.css`: global styles.

### `src/agent` (backend core)

- `bootstrap/`: runtime creation and config loading.
- `bootstrap/runtime_factory.py`: main wiring (settings, db, providers, skills, runtime).
- `bootstrap/settings.py`: dataclasses + config loader.
- `bootstrap/profiles.yaml`: configuration profiles.
- `application/`: use-case orchestration.
- `application/runtime.py`: main agent loop (LLM, tools, memory).
- `application/tool_runtime.py`: safe tool execution (validation, timeout).
- `domain/`: pure business rules.
- `domain/planner.py`: tool policy/planner.
- `domain/memory/*`: memory models and pure scoring/context logic.
- `infrastructure/`: technical adapters.
- `infrastructure/db/*`: SQLite connection, healthchecks, SQL migrations.
- `infrastructure/providers/*`: LLM and embedding integrations.
- `infrastructure/skills/*`: skill registry + builtins implementation.
- `infrastructure/memory/*`: memory retrieval/store/distillation pipeline.
- `infrastructure/tracing.py`: JSON tracing.
- `cli/main.py`: local CLI mode for the agent.

### `tests`

- `tests/backend/test_chat_stream.py`: chat SSE stream tests.
- `tests/backend/test_sse_format.py`: SSE format tests.
- `tests/test_runtime_memory_background.py`: runtime + background memory job tests.
- `tests/test_lt_distill_pipeline.py`: long-term memory pipeline tests.

## Prerequisites

- Python 3.10+
- Node.js 18+

## Installation

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

3. Copy environment configuration:

```bash
cp .env.example .env
```

4. Install Node.js dependencies:

```bash
npm install
npm --prefix apps/web install
```

## Run the agent

### Full mode (API + frontend)

```bash
npm run dev
```

- API: `http://127.0.0.1:8000`
- Web: `http://127.0.0.1:5173`

### API only

```bash
npm run dev:api
```

### Frontend only

```bash
npm run dev:web
```

### CLI mode (without web)

```bash
python -m agent.cli.main
```

## Tests

```bash
python -m unittest discover -s tests -p "test*.py"
```

## Docs

- Memory: [docs/memory/README.md](docs/memory/README.md)
