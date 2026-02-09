# Skill System

The skill system is a plugin-style architecture — similar to GitHub Copilot / Cursor Rules / Claude AGENTS.md — that allows agents to dynamically discover, load, and execute specific instructions and scripts.

---

## 1. File Change Summary

### New Files

| Path | Description |
|------|-------------|
| `src/skills/__init__.py` | Module entry point, exports core components |
| `src/skills/types.py` | Type definitions: `Skill`, `SkillMetadata`, `SkillContent`, `Resource` |
| `src/skills/registry.py` | Skill registry with multi-directory scanning and hierarchical override |
| `src/skills/loader.py` | Progressive loader, loads skill content on demand |
| `src/skills/matcher.py` | Semantic matcher with keyword / path / semantic triggers |
| `src/skills/resources.py` | Resource resolver and script executor |
| `src/skills/toolkit.py` | 6 LangChain tools + `SkillSystem` facade class |
| `src/skills/catalog/git_operations/` | Example skill (Git operations guide) |
| `tests/skills/test_skill_system.py` | Unit tests |

### Modified Files

| Path | Description |
|------|-------------|
| `src/streamlit_app.py` | Sidebar: added "Skills" popover showing skill load status |

---

## 2. Core Architecture

```
Startup                             Runtime
   │                                  │
   ▼                                  ▼
┌──────────────┐    query    ┌──────────────┐
│ SkillRegistry│◄────────────│ SkillMatcher │
│  (metadata)  │             │  (semantic)  │
└──────┬───────┘             └──────────────┘
       │                              │
       ▼ load_content()               │
┌──────────────┐                      │
│ SkillLoader  │◄─────────────────────┘
│  (lazy load) │
└──────────────┘
```

### Hierarchical Override

Priority from low to high:

1. **Global skills**: `src/skills/catalog/`
2. **Agent-specific**: `src/agents/{agent_name}/skills/`

Skills scanned later override earlier ones with the same ID.

---

## 3. Skill Format (SKILL.md)

```yaml
---
name: Skill Name
description: What this skill does
version: "1.0"
triggers:
  - type: semantic
    keywords: ["keyword1", "keyword2"]
  - type: path_glob
    pattern: "**/*.py"
  - type: always
resources:
  - scripts/check.py
  - templates/doc.md
---

# Instruction content
```

---

## 4. Agent Integration

### Option 1: Inject all tools

```python
from skills import skill_tools

agent = create_agent(
    model=model,
    tools=existing_tools + skill_tools,
)
```

### Option 2: Select specific tools

```python
from skills.toolkit import list_available_skills, load_skill

agent_tools = [list_available_skills, load_skill]
```

### Initialize the skill system

```python
from pathlib import Path
from skills import init_skill_system

# Call at service startup
init_skill_system(
    global_root=Path("src/skills/catalog"),
    agent_root=Path("src/agents/my-agent/skills"),  # Optional
)
```

---

## 5. Available Tools

| Tool | Description |
|------|-------------|
| `list_available_skills` | List all available skills and their status |
| `find_skills_for_task` | Find relevant skills based on a task description |
| `load_skill` | Load the full content of a specific skill |
| `get_skill_resource` | Read a resource file from a skill directory |
| `list_skill_resources` | List all resources for a skill |
| `run_skill_script` | Execute a script from a skill directory |

---

## 6. Future Roadmap

### Short-term

1. **Service startup integration**
   Call `init_skill_system()` in `src/service/service.py`'s `lifespan` to auto-initialize skills at startup.

2. **Add tools to specific agents**
   Inject `skill_tools` into `research-assistant` or other agents so they can autonomously query and use skills.

3. **Create more example skills**
   - `code_review`: Code review best practices
   - `testing`: Test writing guidelines
   - `documentation`: Documentation standards

### Medium-term

1. **Enhanced semantic matching**
   Integrate an embedding model (e.g. `sentence-transformers`) to make `find_skills_for_task` smarter.

2. **Dynamic skill loading API**
   Add REST API endpoints to register/unregister skills at runtime.

### Long-term

1. **Skill marketplace**
   Support downloading and installing community skill packs from remote repositories.

2. **Security sandbox**
   Add Docker/subprocess sandboxing for `ScriptExecutor` to isolate script execution.

---

## 7. Design Principles (Industry References)

### Convention over Configuration

File names and locations are fixed (e.g. `AGENTS.md`, `.cursorrules`) — no explicit registration needed.
The system auto-scans `catalog/` directories for `SKILL.md` files.

### Hierarchical Override

OpenAI Codex: `~/.codex/AGENTS.md` → `repo/.codex/AGENTS.md` → `cwd/AGENTS.md`, with closer directories taking higher priority.
This implementation supports agent-specific skills in `src/agents/{agent_name}/skills/`.

### Size Limits and Truncation

Codex defaults to a 32KiB cap; this implementation (`MAX_CONTENT_SIZE = 32 * 1024`) matches that.

### Agent Autonomy

The industry trend is toward letting the LLM decide whether it needs a particular instruction, rather than pre-computing similarity.
The `find_skills_for_task` tool lets the agent proactively query the skill catalog when needed.
