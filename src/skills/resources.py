"""Resource resolution and script execution."""

import logging
import subprocess
from pathlib import Path
from typing import Any

from skills.registry import SkillRegistry
from skills.types import Resource

logger = logging.getLogger(__name__)


class ResourceResolver:
    """Resource resolver.

    Handles additional resources (scripts, templates, data files) in skill directories.
    """

    def __init__(self, registry: SkillRegistry):
        """Initialize the resource resolver.

        Args:
            registry: Skill registry.
        """
        self.registry = registry

    def resolve(self, skill_id: str, resource_name: str) -> Resource | None:
        """Resolve a resource path.

        Args:
            skill_id: Skill ID.
            resource_name: Relative path to the resource.

        Returns:
            Resource object, or None if not found.
        """
        skill = self.registry.get(skill_id)
        if not skill:
            return None

        resource_path = skill.path / resource_name
        if not resource_path.exists():
            return None

        # Infer resource type
        suffix = resource_path.suffix.lower()
        if suffix in (".py", ".sh", ".ps1", ".bat", ".cmd"):
            rtype = "script"
        elif suffix in (".md", ".txt", ".j2", ".jinja2", ".template"):
            rtype = "template"
        else:
            rtype = "data"

        return Resource(
            name=resource_name,
            path=resource_path,
            resource_type=rtype,
        )

    def read_resource(self, skill_id: str, resource_name: str) -> str | None:
        """Read the content of a resource.

        Args:
            skill_id: Skill ID.
            resource_name: Relative path to the resource.

        Returns:
            Resource content, or None if not found.
        """
        resource = self.resolve(skill_id, resource_name)
        if resource and resource.path.is_file():
            try:
                return resource.path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Failed to read resource {skill_id}/{resource_name}: {e}")
                return None
        return None

    def list_resources(self, skill_id: str) -> list[Resource]:
        """List all declared resources for a skill.

        Args:
            skill_id: Skill ID.

        Returns:
            List of resources.
        """
        skill = self.registry.get(skill_id)
        if not skill:
            return []

        resources = []
        for res_name in skill.metadata.resources:
            res = self.resolve(skill_id, res_name)
            if res:
                resources.append(res)
        return resources

    def list_all_files(self, skill_id: str) -> list[Resource]:
        """List all files in the skill directory.

        Args:
            skill_id: Skill ID.

        Returns:
            List of all file resources.
        """
        skill = self.registry.get(skill_id)
        if not skill:
            return []

        resources = []
        for file_path in skill.path.rglob("*"):
            if file_path.is_file() and file_path.name != "SKILL.md":
                rel_path = file_path.relative_to(skill.path)
                res = self.resolve(skill_id, str(rel_path))
                if res:
                    resources.append(res)
        return resources


class ScriptExecutor:
    """Script executor.

    Safely executes scripts from skill directories.
    """

    def __init__(
        self,
        allowed_extensions: set[str] | None = None,
        default_timeout: int = 30,
    ):
        """Initialize the script executor.

        Args:
            allowed_extensions: Set of allowed file extensions.
            default_timeout: Default execution timeout in seconds.
        """
        self.allowed = allowed_extensions or {".py", ".sh"}
        self.default_timeout = default_timeout

    def execute(
        self,
        script_path: Path,
        args: list[str] | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute a script and return the result.

        Args:
            script_path: Path to the script.
            args: Script arguments.
            timeout: Timeout in seconds.
            env: Environment variables.

        Returns:
            Result dict with stdout, stderr, returncode, or error.
        """
        if script_path.suffix not in self.allowed:
            return {"error": f"Script type not allowed: {script_path.suffix}"}

        if not script_path.exists():
            return {"error": f"Script not found: {script_path}"}

        timeout = timeout or self.default_timeout

        try:
            # Build command based on script type
            if script_path.suffix == ".py":
                cmd = ["python", str(script_path)] + (args or [])
            elif script_path.suffix == ".sh":
                cmd = ["bash", str(script_path)] + (args or [])
            elif script_path.suffix in (".ps1",):
                cmd = ["powershell", "-File", str(script_path)] + (args or [])
            else:
                cmd = [str(script_path)] + (args or [])

            logger.info(f"Executing script: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=script_path.parent,
                env=env,
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"Script timed out after {timeout}s: {script_path}")
            return {"error": f"Script timed out after {timeout}s"}
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return {"error": str(e)}

    def is_allowed(self, script_path: Path) -> bool:
        """Check if a script is allowed to execute.

        Args:
            script_path: Path to the script.

        Returns:
            Whether the script is allowed.
        """
        return script_path.suffix in self.allowed
