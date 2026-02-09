"""Skill registry - discovers and manages all skills."""

import logging
from pathlib import Path

import yaml

from skills.types import Skill, SkillMetadata

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Skill registry.

    Scans skill directories, parses SKILL.md frontmatter, and registers all skills.
    Supports multi-level directory scanning with hierarchical override
    (later directories override earlier ones with the same skill ID).

    Priority (low to high):
    1. Global skill directory (src/skills/catalog/)
    2. Agent-specific directory (src/agents/{agent}/skills/)
    3. Session-level (dynamically injected via the register method)
    """

    def __init__(self, *skill_roots: Path):
        """Initialize the registry.

        Args:
            skill_roots: Skill directory root paths, ordered by priority (low to high).
        """
        self.skill_roots = list(skill_roots)
        self._skills: dict[str, Skill] = {}

    def scan(self) -> None:
        """Scan all skill directories and load metadata (later entries override earlier ones)."""
        for root in self.skill_roots:
            self._scan_directory(root)

        logger.info(f"Scanned {len(self._skills)} skills from {len(self.skill_roots)} directories")

    def _scan_directory(self, root: Path) -> None:
        """Scan a single directory."""
        if not root.exists():
            logger.debug(f"Skills directory not found: {root}")
            return

        for skill_dir in root.iterdir():
            if skill_dir.is_dir():
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    self._register_skill(skill_dir, skill_md)

    def _register_skill(self, skill_dir: Path, skill_md: Path) -> None:
        """Register a single skill (metadata only)."""
        try:
            content = skill_md.read_text(encoding="utf-8")
            metadata = self._parse_frontmatter(content)

            skill = Skill(
                id=skill_dir.name,
                path=skill_dir,
                metadata=metadata,
                content=None,  # Lazily loaded
                loaded=False,
            )
            self._skills[skill.id] = skill
            logger.debug(f"Registered skill: {skill.id}")

        except Exception as e:
            logger.error(f"Failed to register skill from {skill_dir}: {e}")

    def _parse_frontmatter(self, content: str) -> SkillMetadata:
        """Parse YAML frontmatter from a SKILL.md file.

        Args:
            content: Raw SKILL.md file content.

        Returns:
            Parsed SkillMetadata.
        """
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                    return SkillMetadata(
                        name=meta.get("name", "Unknown"),
                        description=meta.get("description", ""),
                        version=meta.get("version", "1.0"),
                        triggers=meta.get("triggers", []),
                        resources=meta.get("resources", []),
                        dependencies=meta.get("dependencies", []),
                    )
                except yaml.YAMLError as e:
                    logger.warning(f"Failed to parse YAML frontmatter: {e}")

        return SkillMetadata(name="Unknown", description="")

    def get(self, skill_id: str) -> Skill | None:
        """Get a skill by ID.

        Args:
            skill_id: Skill ID.

        Returns:
            The Skill object, or None if not found.
        """
        return self._skills.get(skill_id)

    def list_all(self) -> list[Skill]:
        """List all registered skills."""
        return list(self._skills.values())

    def list_metadata(self) -> list[SkillMetadata]:
        """Return only the metadata list (lightweight)."""
        return [s.metadata for s in self._skills.values()]

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._skills
