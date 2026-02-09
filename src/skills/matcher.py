"""Skill semantic matcher."""

import fnmatch
import logging
from collections.abc import Callable

from skills.registry import SkillRegistry
from skills.types import Skill

logger = logging.getLogger(__name__)


class SkillMatcher:
    """Skill semantic matcher.

    Finds matching skills based on task descriptions, file paths, etc.
    Supports keyword matching, path glob matching, and extensible semantic scoring.
    """

    def __init__(self, registry: SkillRegistry):
        """Initialize the matcher.

        Args:
            registry: Skill registry.
        """
        self.registry = registry

    def find_matching_skills(
        self,
        query: str,
        context_paths: list[str] | None = None,
        semantic_scorer: Callable[[str, str], float] | None = None,
        min_score: float = 0.1,
    ) -> list[tuple[str, float]]:
        """Find skills matching a query.

        Args:
            query: User query or task description.
            context_paths: File paths involved in the current context.
            semantic_scorer: Optional semantic scoring function (query, skill_desc) -> score.
            min_score: Minimum match score threshold.

        Returns:
            [(skill_id, score), ...] sorted by score descending.
        """
        matches: list[tuple[str, float]] = []

        for skill in self.registry.list_all():
            score = self._calculate_match_score(skill, query, context_paths, semantic_scorer)
            if score >= min_score:
                matches.append((skill.id, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Found {len(matches)} matching skills for query: {query[:50]}...")
        return matches

    def _calculate_match_score(
        self,
        skill: Skill,
        query: str,
        paths: list[str] | None,
        semantic_scorer: Callable[[str, str], float] | None,
    ) -> float:
        """Calculate the match score for a single skill.

        Args:
            skill: Skill object.
            query: Query string.
            paths: File path list.
            semantic_scorer: Semantic scoring function.

        Returns:
            Match score (0.0+, higher is better).
        """
        score = 0.0
        query_lower = query.lower()

        for trigger in skill.metadata.triggers:
            trigger_type = trigger.get("type", "manual")

            if trigger_type == "always":
                score += 1.0

            elif trigger_type == "semantic":
                # Keyword matching
                keywords = trigger.get("keywords", [])
                for kw in keywords:
                    if kw.lower() in query_lower:
                        score += 0.5

                # Use external semantic scorer (e.g. embedding similarity)
                if semantic_scorer and skill.metadata.description:
                    try:
                        sem_score = semantic_scorer(query, skill.metadata.description)
                        score += sem_score * 0.8
                    except Exception as e:
                        logger.warning(f"Semantic scorer failed: {e}")

            elif trigger_type == "path_glob" and paths:
                # Path glob matching
                pattern = trigger.get("pattern", "")
                for path in paths:
                    normalized_path = path.replace("\\", "/")
                    if fnmatch.fnmatch(normalized_path, pattern):
                        score += 0.6
                        break  # One path match is enough

        return score

    def get_always_active_skills(self) -> list[Skill]:
        """Get all skills with an "always" trigger.

        Returns:
            List of always-active skills.
        """
        result = []
        for skill in self.registry.list_all():
            for trigger in skill.metadata.triggers:
                if trigger.get("type") == "always":
                    result.append(skill)
                    break
        return result

    def get_skills_for_path(self, file_path: str) -> list[Skill]:
        """Get skills matching a given file path.

        Args:
            file_path: File path to match.

        Returns:
            List of matching skills.
        """
        result = []
        normalized_path = file_path.replace("\\", "/")

        for skill in self.registry.list_all():
            for trigger in skill.metadata.triggers:
                if trigger.get("type") == "path_glob":
                    pattern = trigger.get("pattern", "")
                    if fnmatch.fnmatch(normalized_path, pattern):
                        result.append(skill)
                        break
        return result
