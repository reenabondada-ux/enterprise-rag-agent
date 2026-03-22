# Infrastructure Setup

## Ollama (Local LLM)

### Install
- macOS (Homebrew): `brew install ollama`

### Start
- `ollama serve`
- Or run a model (starts server implicitly): `ollama run llama3.1:8b`
- If necessary pull the required model: `ollama pull llama3.1:8b`

### Environment
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_CHAT_MODEL` (default: `llama3.1:8b`)

### Stop
- In the running terminal: `Ctrl+C`
- Or: `pkill ollama`

## Docker Compose

### Start services
- From repo root: `docker compose -f infra/docker-compose.yml up -d`

### View logs
- `docker compose -f infra/docker-compose.yml logs -f`

### Stop services
- `docker compose -f infra/docker-compose.yml down`
Add option -v to take down the volumes or delete the tables
- `docker compose -f infra/docker-compose.yml down -v`