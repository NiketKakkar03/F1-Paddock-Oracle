"""Single deploy entry point — registers both functions under one app.

Deploy with:
    modal deploy modal_backend/deploy.py
"""

from modal_backend.commentary import app, generate_commentary, persona_chat  # noqa: F401
from modal_backend.strategy import reason_strategy  # noqa: F401
