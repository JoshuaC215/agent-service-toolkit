"""Progressive skill loader."""

import logging

from skills.registry import SkillRegistry
from skills.types import Skill, SkillContent

logger = logging.getLogger(__name__)

# Maximum content size per skill in bytes.
MAX_CONTENT_SIZE = 32 * 1024  # 32KB


class SkillLoader:
    """Progressive skill loader.

    Loads full skill content on demand, with caching and unloading support.
    """

    def __init__(self, registry: SkillRegistry):
        """Initialize the loader.

        Args:
            registry: Skill registry.
        """
        self.registry = registry
        self._content_cache: dict[str, SkillContent] = {}

    def load_content(self, skill_id: str) -> SkillContent | None:
        """Load the full content of a skill on demand.

        Args:
            skill_id: Skill ID.

        Returns:
            Skill content, or None if not found.
        """
        # Check cache
        if skill_id in self._content_cache:
            logger.debug(f"Returning cached content for skill: {skill_id}")
            return self._content_cache[skill_id]

        skill = self.registry.get(skill_id)
        if not skill:
            logger.warning(f"Skill not found: {skill_id}")
            return None

        skill_md = skill.path / "SKILL.md"
        if not skill_md.exists():
            logger.warning(f"SKILL.md not found: {skill_md}")
            return None

        try:
            raw = skill_md.read_text(encoding="utf-8")
            content = self._parse_content(raw)

            # Truncate overly long content
            if len(content.instructions) > MAX_CONTENT_SIZE:
                content.instructions = content.instructions[:MAX_CONTENT_SIZE] + "\n...[truncated]"
                logger.warning(f"Skill content truncated: {skill_id}")

            # Update cache and state
            self._content_cache[skill_id] = content
            skill.content = content
            skill.loaded = True

            logger.info(f"Loaded skill content: {skill_id} ({len(content.instructions)} bytes)")
            return content

        except Exception as e:
            logger.error(f"Failed to load skill content for {skill_id}: {e}")
            return None

    def _parse_content(self, raw: str) -> SkillContent:
        """Parse SKILL.md content (skip frontmatter).

        Args:
            raw: Raw file content.

        Returns:
            Parsed SkillContent.
        """
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            body = parts[2].strip() if len(parts) >= 3 else ""
        else:
            body = raw.strip()

        return SkillContent(instructions=body)

    def unload(self, skill_id: str) -> None:
        """Unload skill content (free memory).

        Args:
            skill_id: Skill ID.
        """
        if skill_id in self._content_cache:
            del self._content_cache[skill_id]
            logger.debug(f"Unloaded skill content from cache: {skill_id}")

        skill = self.registry.get(skill_id)
        if skill:
            skill.content = None
            skill.loaded = False

    def unload_all(self) -> None:
        """Unload all skill content."""
        self._content_cache.clear()
        for skill in self.registry.list_all():
            skill.content = None
            skill.loaded = False
        logger.info("Unloaded all skill contents")

    def is_loaded(self, skill_id: str) -> bool:
        """Check if a skill's content has been loaded.

        Args:
            skill_id: Skill ID.

        Returns:
            Whether the full content is loaded.
        """
        return skill_id in self._content_cache

    def get_loaded_skills(self) -> list[Skill]:
        """Get all skills whose content has been loaded."""
        return [skill for skill in self.registry.list_all() if skill.loaded]
