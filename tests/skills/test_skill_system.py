"""Skill system unit tests."""

from pathlib import Path

import pytest

from skills.loader import SkillLoader
from skills.matcher import SkillMatcher
from skills.registry import SkillRegistry
from skills.resources import ResourceResolver, ScriptExecutor
from skills.toolkit import SkillSystem, get_skill_system, init_skill_system
from skills.types import SkillMetadata


@pytest.fixture
def sample_skills_dir(tmp_path: Path) -> Path:
    """Create a sample skills directory for testing."""
    # Create test_skill
    skill_dir = tmp_path / "test_skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: Test Skill
description: A test skill for unit testing
version: "1.0"
triggers:
  - type: semantic
    keywords: ["test", "example", "demo"]
  - type: path_glob
    pattern: "**/*.test.py"
resources:
  - scripts/helper.py
---

# Test Skill Instructions

This is the instruction content for testing.

## Usage

Use this skill when you need to test something.
""",
        encoding="utf-8",
    )

    # Create scripts directory and file
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "helper.py").write_text(
        """#!/usr/bin/env python3
print("Hello from helper script!")
""",
        encoding="utf-8",
    )

    # Create always_skill
    always_dir = tmp_path / "always_skill"
    always_dir.mkdir()
    (always_dir / "SKILL.md").write_text(
        """---
name: Always Active Skill
description: A skill that is always active
triggers:
  - type: always
---

# Always Active

This skill is always loaded.
""",
        encoding="utf-8",
    )

    return tmp_path


class TestSkillRegistry:
    """Test the skill registry."""

    def test_scan_discovers_skills(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        assert len(registry) == 2
        assert "test_skill" in registry
        assert "always_skill" in registry

    def test_get_skill(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()

        skill = registry.get("test_skill")
        assert skill is not None
        assert skill.metadata.name == "Test Skill"
        assert skill.metadata.version == "1.0"
        assert len(skill.metadata.triggers) == 2

    def test_list_all(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()

        skills = registry.list_all()
        assert len(skills) == 2

    def test_list_metadata(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()

        metadata_list = registry.list_metadata()
        assert len(metadata_list) == 2
        assert all(isinstance(m, SkillMetadata) for m in metadata_list)

    def test_empty_directory(self, tmp_path: Path):
        registry = SkillRegistry(tmp_path)
        registry.scan()
        assert len(registry) == 0

    def test_nonexistent_directory(self, tmp_path: Path):
        registry = SkillRegistry(tmp_path / "nonexistent")
        registry.scan()
        assert len(registry) == 0


class TestSkillLoader:
    """Test the skill loader."""

    def test_load_content(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        loader = SkillLoader(registry)

        content = loader.load_content("test_skill")
        assert content is not None
        assert "Test Skill Instructions" in content.instructions

    def test_lazy_loading(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        loader = SkillLoader(registry)

        skill = registry.get("test_skill")
        assert not skill.loaded
        assert skill.content is None

        loader.load_content("test_skill")
        assert skill.loaded
        assert skill.content is not None

    def test_cache(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        loader = SkillLoader(registry)

        content1 = loader.load_content("test_skill")
        content2 = loader.load_content("test_skill")
        assert content1 is content2

    def test_unload(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        loader = SkillLoader(registry)

        loader.load_content("test_skill")
        assert loader.is_loaded("test_skill")

        loader.unload("test_skill")
        assert not loader.is_loaded("test_skill")

    def test_nonexistent_skill(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        loader = SkillLoader(registry)

        content = loader.load_content("nonexistent")
        assert content is None


class TestSkillMatcher:
    """Test the skill matcher."""

    def test_semantic_matching(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        matcher = SkillMatcher(registry)

        matches = matcher.find_matching_skills("I need to test something")
        assert len(matches) > 0
        skill_ids = [m[0] for m in matches]
        assert "test_skill" in skill_ids

    def test_path_matching(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        matcher = SkillMatcher(registry)

        matches = matcher.find_matching_skills(
            "run some code",
            context_paths=["src/utils/helper.test.py"],
        )
        skill_ids = [m[0] for m in matches]
        assert "test_skill" in skill_ids

    def test_always_trigger(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        matcher = SkillMatcher(registry)

        always_skills = matcher.get_always_active_skills()
        assert len(always_skills) == 1
        assert always_skills[0].id == "always_skill"

    def test_no_matches(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        matcher = SkillMatcher(registry)

        matches = matcher.find_matching_skills("completely unrelated query xyz123")
        # always_skill will match because it has an "always" trigger
        skill_ids = [m[0] for m in matches]
        assert "always_skill" in skill_ids


class TestResourceResolver:
    """Test the resource resolver."""

    def test_resolve_script(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        resolver = ResourceResolver(registry)

        resource = resolver.resolve("test_skill", "scripts/helper.py")
        assert resource is not None
        assert resource.resource_type == "script"

    def test_read_resource(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        resolver = ResourceResolver(registry)

        content = resolver.read_resource("test_skill", "scripts/helper.py")
        assert content is not None
        assert "Hello from helper script" in content

    def test_list_resources(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        resolver = ResourceResolver(registry)

        resources = resolver.list_resources("test_skill")
        assert len(resources) == 1
        assert resources[0].name == "scripts/helper.py"

    def test_nonexistent_resource(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()
        resolver = ResourceResolver(registry)

        resource = resolver.resolve("test_skill", "nonexistent.py")
        assert resource is None


class TestScriptExecutor:
    """Test the script executor."""

    def test_execute_python_script(self, sample_skills_dir: Path):
        registry = SkillRegistry(sample_skills_dir)
        registry.scan()

        script_path = sample_skills_dir / "test_skill" / "scripts" / "helper.py"
        executor = ScriptExecutor()

        result = executor.execute(script_path)
        assert "error" not in result
        assert "Hello from helper script" in result["stdout"]
        assert result["returncode"] == 0

    def test_disallowed_extension(self, sample_skills_dir: Path):
        executor = ScriptExecutor()
        fake_path = sample_skills_dir / "test.exe"
        fake_path.touch()

        result = executor.execute(fake_path)
        assert "error" in result
        assert "not allowed" in result["error"]

    def test_is_allowed(self):
        executor = ScriptExecutor(allowed_extensions={".py", ".sh"})
        assert executor.is_allowed(Path("test.py"))
        assert executor.is_allowed(Path("test.sh"))
        assert not executor.is_allowed(Path("test.exe"))


class TestSkillSystem:
    """Test the skill system facade."""

    def test_init(self, sample_skills_dir: Path):
        system = SkillSystem(sample_skills_dir)
        assert len(system.registry) == 2
        assert system.loader is not None
        assert system.matcher is not None
        assert system.resolver is not None

    def test_global_instance(self, sample_skills_dir: Path):
        system = init_skill_system(sample_skills_dir)
        assert get_skill_system() is system

    def test_reload(self, sample_skills_dir: Path):
        system = SkillSystem(sample_skills_dir)
        system.loader.load_content("test_skill")
        assert system.loader.is_loaded("test_skill")

        system.reload()
        assert not system.loader.is_loaded("test_skill")
