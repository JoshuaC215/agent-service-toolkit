"""Skill System

Provides intelligent skill discovery, progressive loading, and semantic matching.
"""

from skills.loader import SkillLoader
from skills.matcher import SkillMatcher
from skills.registry import SkillRegistry
from skills.resources import ResourceResolver, ScriptExecutor
from skills.toolkit import SkillSystem, init_skill_system, skill_tools
from skills.types import Resource, Skill, SkillContent, SkillMetadata, TriggerType

__all__ = [
    # Types
    "Skill",
    "SkillMetadata",
    "SkillContent",
    "Resource",
    "TriggerType",
    # Core
    "SkillRegistry",
    "SkillLoader",
    "SkillMatcher",
    "ResourceResolver",
    "ScriptExecutor",
    # Toolkit
    "skill_tools",
    "init_skill_system",
    "SkillSystem",
]
