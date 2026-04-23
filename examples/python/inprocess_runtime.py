"""Minimal in-process example for the live next runtime.

Run from the repo root:

    python examples/python/inprocess_runtime.py \
      "Compare the auth and rate limit docs." \
      my-chat-001
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_chatbot_next.config import load_settings
from agentic_chatbot_next.providers import build_providers
from agentic_chatbot_next.app.service import RuntimeService


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else "What does the API auth doc say?"
    conversation_id = sys.argv[2] if len(sys.argv) > 2 else "inprocess-demo"

    settings = load_settings()
    providers = build_providers(settings)
    service = RuntimeService.create(settings, providers)
    session = RuntimeService.create_local_session(settings, conversation_id=conversation_id)

    answer = service.process_turn(session, user_text=question)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
