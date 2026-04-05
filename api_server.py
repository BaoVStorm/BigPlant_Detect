#!/usr/bin/env python3

"""
Backward-compatible entrypoint.

Keep existing run command:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

All API routes, config, and backend selection now live in app.main.
"""

from app.main import app
