"""Skill toolkit - Agent-callable skill tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from langchain_core.tools import tool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Global skill system instance
_skill_system: SkillSystem | None = None


class SkillSystem:
    """Skill system facade.

    Integrates all skill-related components and provides a unified interface.
    Supports multi-level directory scanning with hierarchical override.
    """

    def __init__(self, *skill_roots: Path):
        """Initialize the skill system.

        Args:
            *skill_roots: Skill directory root paths, ordered by priority (low to high).
        """
        from skills.loader import SkillLoader
        from skills.matcher import SkillMatcher
        from skills.registry import SkillRegistry
        from skills.resources import ResourceResolver, ScriptExecutor

        self.skill_roots = list(skill_roots)
        self.registry = SkillRegistry(*skill_roots)
        self.loader = SkillLoader(self.registry)
        self.matcher = SkillMatcher(self.registry)
        self.resolver = ResourceResolver(self.registry)
        self.executor = ScriptExecutor()

        # Scan and load metadata at startup
        self.registry.scan()
        logger.info(
            f"SkillSystem initialized with {len(self.registry)} skills "
            f"from {len(self.skill_roots)} directories"
        )

    def reload(self) -> None:
        """Re-scan and reload skills."""
        self.loader.unload_all()
        self.registry.scan()
        logger.info(f"SkillSystem reloaded with {len(self.registry)} skills")


def init_skill_system(
    global_root: Path,
    agent_root: Path | None = None,
) -> SkillSystem:
    """Initialize the global skill system (call at service startup).

    Args:
        global_root: Global skill catalog root path.
        agent_root: Agent-specific skill directory (optional, higher priority).

    Returns:
        The SkillSystem instance.
    """
    global _skill_system
    roots = [global_root]
    if agent_root and agent_root.exists():
        roots.append(agent_root)
    _skill_system = SkillSystem(*roots)
    return _skill_system


def get_skill_system() -> SkillSystem | None:
    """Get the global skill system instance."""
    return _skill_system


# ============================================================
# LangChain tool definitions
# ============================================================


@tool
def list_available_skills() -> str:
    """List all available skills and their descriptions.

    Use this to discover what skills are available to help with the current task.
    Returns a list of skills including skill ID, load status, and description.
    """
    if not _skill_system:
        return "Skill system not initialized"

    skills = _skill_system.registry.list_all()
    if not skills:
        return "No skills available"

    lines = ["## Available Skills\n"]
    for s in skills:
        status = "✓ loaded" if s.loaded else "○ not loaded"
        lines.append(f"- **{s.id}** [{status}]")
        lines.append(f"  - {s.metadata.description}")
    return "\n".join(lines)


@tool
def find_skills_for_task(
    task_description: Annotated[str, "Description of the current task"],
    file_paths: Annotated[list[str] | None, "List of involved file paths (optional)"] = None,
) -> str:
    """Find skills relevant to a task description.

    Analyzes the task description and involved file paths, returning skills
    sorted by relevance. Use this before starting a task to identify which
    skill guides to consult.

    Args:
        task_description: Description of the current task.
        file_paths: List of involved file paths (optional).
    """
    if not _skill_system:
        return "Skill system not initialized"

    matches = _skill_system.matcher.find_matching_skills(
        task_description,
        file_paths,
    )

    if not matches:
        return "No skills found matching the task"

    lines = ["## Relevant Skills (by match score)\n"]
    for skill_id, score in matches[:5]:  # Return at most 5
        skill = _skill_system.registry.get(skill_id)
        if skill:
            lines.append(f"- **{skill_id}** (score: {score:.2f})")
            lines.append(f"  - {skill.metadata.description}")
    return "\n".join(lines)


@tool
def load_skill(
    skill_id: Annotated[str, "ID of the skill to load"],
) -> str:
    """Load the full instruction content of a skill.

    Only call this when you are sure you need a particular skill.
    Returns the skill's detailed instructions and best practices.

    Args:
        skill_id: ID of the skill to load.
    """
    if not _skill_system:
        return "Skill system not initialized"

    skill = _skill_system.registry.get(skill_id)
    if not skill:
        return f"Skill not found: {skill_id}"

    content = _skill_system.loader.load_content(skill_id)
    if not content:
        return f"Failed to load skill content: {skill_id}"

    return f"# Skill: {skill.metadata.name}\n\n{content.instructions}"


@tool
def get_skill_resource(
    skill_id: Annotated[str, "Skill ID"],
    resource_name: Annotated[str, "Relative path to the resource, e.g. 'scripts/lint.py'"],
) -> str:
    """Read a resource file from a skill directory.

    Use this to read templates, configuration files, or other auxiliary files.
    The resource path is relative to the skill directory.

    Args:
        skill_id: Skill ID.
        resource_name: Relative path to the resource file.
    """
    if not _skill_system:
        return "Skill system not initialized"

    content = _skill_system.resolver.read_resource(skill_id, resource_name)
    if not content:
        return f"Resource not found: {skill_id}/{resource_name}"
    return content


@tool
def list_skill_resources(
    skill_id: Annotated[str, "Skill ID"],
) -> str:
    """List all resource files in a skill directory.

    Shows scripts, templates, and data files included with the skill.

    Args:
        skill_id: Skill ID.
    """
    if not _skill_system:
        return "Skill system not initialized"

    skill = _skill_system.registry.get(skill_id)
    if not skill:
        return f"Skill not found: {skill_id}"

    resources = _skill_system.resolver.list_all_files(skill_id)
    if not resources:
        return f"Skill {skill_id} has no additional resource files"

    lines = [f"## Resources for skill {skill_id}\n"]
    for res in resources:
        lines.append(f"- [{res.resource_type}] `{res.name}`")
    return "\n".join(lines)


@tool
def run_skill_script(
    skill_id: Annotated[str, "Skill ID"],
    script_name: Annotated[str, "Relative path to the script file"],
    args: Annotated[list[str] | None, "Script arguments (optional)"] = None,
) -> str:
    """Execute a script from a skill directory.

    ⚠️ Use with caution — only call when explicitly needed.
    Scripts must be .py or .sh files (Windows also supports .ps1).

    Args:
        skill_id: Skill ID.
        script_name: Relative path to the script file.
        args: Script arguments (optional).
    """
    if not _skill_system:
        return "Skill system not initialized"

    resource = _skill_system.resolver.resolve(skill_id, script_name)
    if not resource:
        return f"Script not found: {skill_id}/{script_name}"

    if resource.resource_type != "script":
        return f"Resource is not a script: {skill_id}/{script_name}"

    if not _skill_system.executor.is_allowed(resource.path):
        return f"Script type not allowed: {resource.path.suffix}"

    result = _skill_system.executor.execute(resource.path, args)

    if "error" in result:
        return f"Execution error: {result['error']}"

    output_parts = []
    if result.get("stdout"):
        output_parts.append(f"**Output:**\n```\n{result['stdout']}\n```")
    if result.get("stderr"):
        output_parts.append(f"**Stderr:**\n```\n{result['stderr']}\n```")
    output_parts.append(f"**Exit code:** {result.get('returncode', 'unknown')}")

    return "\n\n".join(output_parts) if output_parts else "(no output)"


# Export all skill tools
skill_tools = [
    list_available_skills,
    find_skills_for_task,
    load_skill,
    get_skill_resource,
    list_skill_resources,
    run_skill_script,
]
