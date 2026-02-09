"""Core type definitions for the skill system."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TriggerType(Enum):
    """Trigger type enumeration."""

    ALWAYS = "always"  # Always loaded
    MANUAL = "manual"  # Manually triggered
    SEMANTIC = "semantic"  # Semantic match trigger
    PATH_GLOB = "path_glob"  # Path glob match trigger


@dataclass
class SkillMetadata:
    """Skill metadata (lightweight, loaded at startup).

    Attributes:
        name: Skill name.
        description: Skill description.
        version: Version string.
        triggers: List of trigger rules.
        resources: List of resource file paths.
        dependencies: List of dependent skill IDs.
    """

    name: str
    description: str
    version: str = "1.0"
    triggers: list[dict] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class SkillContent:
    """Full skill content (loaded on demand).

    Attributes:
        instructions: Primary instruction content.
        examples: List of usage examples.
    """

    instructions: str
    examples: list[str] = field(default_factory=list)


@dataclass
class Skill:
    """Complete skill object.

    Attributes:
        id: Unique identifier, e.g. "git_operations".
        path: Directory containing SKILL.md.
        metadata: Skill metadata.
        content: Full content (lazily loaded).
        loaded: Whether full content has been loaded.
    """

    id: str
    path: Path
    metadata: SkillMetadata
    content: SkillContent | None = None
    loaded: bool = False


@dataclass
class Resource:
    """Resource reference.

    Attributes:
        name: Resource name.
        path: Full path to the resource.
        resource_type: Resource type (script/template/data).
    """

    name: str
    path: Path
    resource_type: str  # "script" | "template" | "data"
