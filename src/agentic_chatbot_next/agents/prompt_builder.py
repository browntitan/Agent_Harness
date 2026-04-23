from __future__ import annotations

from pathlib import Path


class PromptBuilder:
    """Minimal prompt resolver for markdown-backed agent prompts."""

    def __init__(self, skills_dir: Path, *, overlay_dir: Path | None = None):
        self.skills_dir = skills_dir
        self.overlay_dir = overlay_dir

    def _resolve_path(self, prompt_file: str) -> Path:
        if self.overlay_dir is not None:
            overlay_path = self.overlay_dir / prompt_file
            if overlay_path.exists():
                return overlay_path
        path = self.skills_dir / prompt_file
        if not path.exists():
            raise FileNotFoundError(f"Prompt file {prompt_file!r} was not found under {self.skills_dir}.")
        return path

    def load_prompt(self, prompt_file: str) -> str:
        return self._resolve_path(prompt_file).read_text(encoding="utf-8")

    def load_shared_prompt(self, prompt_file: str = "skills.md") -> str:
        try:
            return self.load_prompt(prompt_file)
        except FileNotFoundError:
            return ""
