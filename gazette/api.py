"""ASGI entry point: ``uvicorn gazette.api:app``."""

from gazette.interfaces.http import create_app

app = create_app()
