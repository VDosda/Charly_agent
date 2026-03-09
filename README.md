# Charly_agent

Agent backend + API FastAPI + frontend React.

## Architecture

- API HTTP: `apps/api`
- Frontend web: `apps/web`
- Coeur agent: `src/agent`

## Prerequis

- Python 3.10+
- Node.js 18+

## Installation

1. Creer et activer un environnement virtuel Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installer les dependances Python:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

3. Copier la config d'environnement:

```bash
cp .env.example .env
```

4. Installer les dependances Node.js:

```bash
npm install
npm --prefix apps/web install
```

## Lancer l'agent

### Mode complet (API + frontend)

```bash
npm run dev
```

- API: `http://127.0.0.1:8000`
- Web: `http://127.0.0.1:5173`

### Lancer seulement l'API

```bash
npm run dev:api
```

### Lancer seulement le frontend

```bash
npm run dev:web
```

### Mode CLI (sans web)

```bash
python -m agent.cli.main
```

## Tests

```bash
python -m unittest discover -s tests -p "test*.py"
```

## Docs

- Memory: [docs/memory/README.md](docs/memory/README.md)
