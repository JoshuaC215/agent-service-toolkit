# Task Queue Design Spec — V0.1

> **Status**: Draft / RFC  
> **Author**: @xtaq  
> **Discussion**: [#266](https://github.com/JoshuaC215/agent-service-toolkit/discussions/266)  
> **Scope**: Minimal in-process task queue for handling burst requests without external dependencies.

---

## Problem

When multiple users submit agent tasks concurrently, the service processes them all immediately. Under burst load this causes:

- Memory/CPU spikes from parallel LLM calls
- Timeout failures when too many tasks compete for resources
- No visibility into how many tasks are waiting or how long they'll wait

## Goals (V0.1)

1. **Bounded concurrency** — limit parallel agent executions to a configurable max
2. **FIFO fairness** — first submitted, first executed
3. **Visibility** — callers can check position-in-queue and task state
4. **Zero new dependencies** — `asyncio.Queue` only, no Redis/Celery/external broker

## Non-Goals (V0.1)

- Per-user quotas or rate limiting (→ V0.2)
- Priority tiers (→ V0.2)
- Persistent queue (survives restart) (→ V0.2+, likely Redis-backed)
- Cost attribution per task (→ V0.2+)

---

## API Surface

### `POST /queue/submit`

Submit a task to the queue. Returns immediately with a task handle.

**Request body** — same as the existing invoke/stream endpoint body, plus optional metadata:

```json
{
  "agent_id": "research-assistant",
  "input": { "messages": [...] },
  "metadata": {}
}
```

**Response** (`202 Accepted`):

```json
{
  "task_id": "t_abc123",
  "status": "queued",
  "position": 3,
  "submitted_at": "2026-04-06T10:00:00Z"
}
```

### `GET /queue/status/{task_id}`

Poll task state. Lightweight, no side effects.

**Response**:

```json
{
  "task_id": "t_abc123",
  "status": "queued | running | done | failed",
  "position": 3,
  "result_url": "/queue/result/t_abc123",
  "submitted_at": "2026-04-06T10:00:00Z",
  "started_at": null,
  "completed_at": null
}
```

- `position` is `null` when status is `running | done | failed`
- `result_url` is `null` until status is `done`

### `GET /queue/result/{task_id}`

Retrieve the completed task output. Returns `404` if not done yet.

**Response**: Same shape as the existing agent response body.

### `GET /queue/stats`

Global queue health. No auth required (useful for monitoring dashboards).

**Response**:

```json
{
  "queue_depth": 7,
  "active_workers": 3,
  "max_workers": 3,
  "avg_wait_ms": 4200,
  "avg_execution_ms": 12500,
  "tasks_completed_last_hour": 42
}
```

---

## Internal Design

### Core Components

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  HTTP Layer  │────▶│  TaskQueue   │────▶│   Workers   │
│  (FastAPI)   │     │ (asyncio.Q)  │     │ (async pool)│
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────────────┐
                    │  TaskStore   │
                    │ (in-memory)  │
                    └──────────────┘
```

### `TaskQueue`

```python
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import uuid


class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str = field(default_factory=lambda: f"t_{uuid.uuid4().hex[:12]}")
    agent_id: str = ""
    input: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.QUEUED
    position: int | None = None
    result: dict | None = None
    error: str | None = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None


class TaskQueue:
    def __init__(self, max_workers: int = 3):
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._store: dict[str, Task] = {}
        self._max_workers = max_workers
        self._active_workers = 0
        self._completed_count = 0

    async def submit(self, agent_id: str, input: dict, metadata: dict | None = None) -> Task:
        task = Task(agent_id=agent_id, input=input, metadata=metadata or {})
        self._store[task.task_id] = task
        await self._queue.put(task.task_id)
        task.position = self._queue.qsize()
        return task

    async def worker(self, execute_fn):
        """Long-running worker coroutine. Pulls tasks from queue and executes."""
        while True:
            task_id = await self._queue.get()
            task = self._store.get(task_id)
            if not task:
                self._queue.task_done()
                continue
            task.status = TaskStatus.RUNNING
            task.position = None
            task.started_at = datetime.now(timezone.utc)
            self._active_workers += 1
            try:
                result = await execute_fn(task.agent_id, task.input)
                task.result = result
                task.status = TaskStatus.DONE
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
            finally:
                task.completed_at = datetime.now(timezone.utc)
                self._active_workers -= 1
                self._completed_count += 1
                self._queue.task_done()

    def get_task(self, task_id: str) -> Task | None:
        return self._store.get(task_id)

    def stats(self) -> dict:
        return {
            "queue_depth": self._queue.qsize(),
            "active_workers": self._active_workers,
            "max_workers": self._max_workers,
            "tasks_completed": self._completed_count,
        }
```

### Startup Integration

In `run_service.py` or equivalent:

```python
task_queue = TaskQueue(max_workers=settings.queue_max_workers)  # default 3

@app.on_event("startup")
async def start_workers():
    for _ in range(task_queue._max_workers):
        asyncio.create_task(task_queue.worker(execute_fn=run_agent))
```

### Configuration

Single env var to start:

```bash
QUEUE_MAX_WORKERS=3  # default, configurable
```

---

## Migration Path

| Version | What Changes | Dependencies |
|---------|-------------|--------------|
| **V0.1** (this spec) | In-memory asyncio queue, 4 endpoints | None (stdlib only) |
| **V0.2** | Redis backend option, per-user tracking, priority | `redis` (optional) |
| **V0.3** | Cost attribution per task, billing hooks | Depends on billing design |

The API surface stays identical across versions — only the backend changes.

---

## Open Questions

1. **Result TTL**: How long should completed task results stay in memory? Suggest 1 hour default, configurable.
2. **Queue size limit**: Should we cap the queue? Suggest `QUEUE_MAX_SIZE=1000` with `429 Too Many Requests` when full.
3. **Streaming support**: Should queued tasks support streaming responses? Could use SSE on `/queue/stream/{task_id}` but adds complexity — recommend deferring to V0.2.

---

## Next Steps

- [ ] Review & feedback on this spec
- [ ] Agree on endpoint paths (proposed: `/queue/*`)
- [ ] Implementation PR (estimated: ~200 LOC for core + routes)
