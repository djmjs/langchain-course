uv venv

source .venv/bin/activate

uv pip freeze > requirements.lock
