from os import getenv

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import logging
import re
import sys
from datetime import UTC, datetime
from io import BytesIO

import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame

# Load .env so subprocess can pick up environment configuration
load_dotenv()
logger = logging.getLogger(__name__)


def langfuse_trace_call(
    public_key: str,
    secret_key: str,
    lf_url: str,
    page: int,
    name: str,
    fromTimestamp: str,
    toTimestamp: str,
):
    response = requests.get(
        f"{lf_url}/api/public/traces",
        auth=(f"{public_key}", f"{secret_key}"),
        params={
            "page": f"{page}",
            "name": f"{name}",
            "fromTimestamp": f"{fromTimestamp}",
            "toTimestamp": f"{toTimestamp}",
        },
        timeout=30,
    )
    try:
        response.raise_for_status()
        return response
    except requests.HTTPError as err:
        status = getattr(err.response, "status_code", None) or getattr(response, "status_code", "")
        try:
            body = (
                err.response.text if getattr(err, "response", None) is not None else response.text
            )[:500]
        except Exception:
            body = "<no body>"
        raise Exception(f"HTTP error {status}: {err}. Body: {body}")
    except Exception as err:
        raise Exception(f"Request failed: {err}")


def get_lf_traces(
    public_key: str,
    secret_key: str,
    lfurl: str,
    name_agent: str,
    fromTimestamp: str,
    toTimestamp: str,
) -> list[dict]:
    page_num = 1
    batch_len = 1
    traces: list[dict] = []
    while batch_len > 0:
        trace = langfuse_trace_call(
            public_key,
            secret_key,
            lfurl,
            page=page_num,
            name=name_agent,
            fromTimestamp=fromTimestamp,
            toTimestamp=toTimestamp,
        )
        data = trace.json().get("data", [])
        traces += data
        page_num += 1
        batch_len = len(data)
    return traces


def _safe_png_filename(agent_name: str, from_ts: str, to_ts: str) -> str:
    def _compact(ts: str) -> str:
        try:
            t = ts[:-1] + "+00:00" if isinstance(ts, str) and ts.endswith("Z") else ts
            dt = datetime.fromisoformat(t)
        except Exception as e:
            raise ValueError(f"Invalid ISO8601 timestamp: {ts}") from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        dt = dt.astimezone(UTC)
        return dt.strftime("%Y%m%d%H%M")

    safe_agent = re.sub(r"[^A-Za-z0-9_-]+", "_", agent_name or "agent")
    return f"skill_levels_{safe_agent}_{_compact(from_ts)}_{_compact(to_ts)}_UTC.png"


def generate_skillcompanion_png_bytes(
    from_timestamp: str,
    to_timestamp: str,
    agent_name: str = "skillcompanion_interrupted",
) -> tuple[bytes, str]:
    """
    Generate the Skill Companion bar chart PNG and return PNG bytes and a suggested filename.
    """
    # Read configuration from environment (.env)
    lf_public_key = getenv("LANGFUSE_PUBLIC_KEY")
    lf_secret_key = getenv("LANGFUSE_SECRET_KEY")
    lf_url = getenv("LANGFUSE_HOST")

    if not lf_public_key or not lf_secret_key or not lf_url:
        raise RuntimeError(
            "Missing required env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST must be set."
        )

    traces = get_lf_traces(
        lf_public_key,
        lf_secret_key,
        lf_url,
        name_agent=agent_name,
        fromTimestamp=from_timestamp,
        toTimestamp=to_timestamp,
    )

    rows: list[dict] = []
    skipped_count = 0

    for i, entry in enumerate(traces):
        if not isinstance(entry, dict):
            skipped_count += 1
            logger.warning("Skipping non-dict trace at index %d (type=%s)", i, type(entry).__name__)
            continue

        try:
            output = entry.get("output")
            category = output.get("category") if isinstance(output, dict) else None
            row = {
                "id": entry.get("id"),
                "timestamp": entry.get("timestamp"),
                "userId": entry.get("userId"),
                "sessionId": entry.get("sessionId"),
                "username": entry.get("input", {}).get("variables", {}).get("{{USER_NAME}}"),
                "output": output,
                "category": category,
            }
            rows.append(row)
        except (TypeError, ValueError, KeyError) as e:
            skipped_count += 1
            logger.warning(
                "Skipping malformed trace at index %d (id=%s): %s",
                i,
                entry.get("id"),
                e,
                exc_info=True,
            )

    if skipped_count:
        logger.info("Finished collecting rows: %d valid, %d skipped", len(rows), skipped_count)

    df = DataFrame(rows)

    if not df.empty and "category" in df.columns:
        beginner_count = df["category"].str.contains(r"Beginner", case=False, na=False).sum()
        advanced_count = df["category"].str.contains(r"Advanced", case=False, na=False).sum()
        expert_count = df["category"].str.contains(r"Expert", case=False, na=False).sum()
    else:
        beginner_count = 0
        advanced_count = 0
        expert_count = 0

    try:
        labels = ["Beginner", "Advanced", "Expert"]
        values = [int(beginner_count), int(advanced_count), int(expert_count)]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.bar(labels, values, color=["#4CAF50", "#2196F3", "#9C27B0"])

        def _fmt_utc(ts: str) -> str:
            try:
                t = ts[:-1] + "+00:00" if isinstance(ts, str) and ts.endswith("Z") else ts
                dt = datetime.fromisoformat(t)
            except Exception as e:
                raise ValueError(f"Invalid ISO8601 timestamp: {ts}") from e
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            dt = dt.astimezone(UTC)
            return dt.strftime("%d.%m.%Y %H:%M")

        title_text = (
            f"Skill Companion Ergebnis – {_fmt_utc(from_timestamp)} – {_fmt_utc(to_timestamp)} UTC"
        )
        ax.set_title(title_text, pad=16)
        ax.set_ylabel("Anzahl")
        upper = max(values) if max(values) > 0 else 1
        ax.set_ylim(0, upper * 1.3)

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.bar_label(bars, labels=[str(v) for v in values], padding=3)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        png_bytes = buf.getvalue()
        filename = _safe_png_filename(agent_name, from_timestamp, to_timestamp)
        return png_bytes, filename
    except ImportError:
        raise RuntimeError(
            "matplotlib is not installed. Install with: 'uv add matplotlib' or 'pip install matplotlib'."
        )
    except Exception as e:
        raise RuntimeError(f"Chart generation failed: {e}")


def generate_skillcompanion_png(
    from_timestamp: str,
    to_timestamp: str,
    agent_name: str = "skillcompanion_interrupted",
    output_path: str | None = None,
) -> str:
    """
    Backwards-compatible wrapper that writes the PNG to disk and returns the path.
    """
    png_bytes, suggested_name = generate_skillcompanion_png_bytes(
        from_timestamp, to_timestamp, agent_name=agent_name
    )
    out_path = output_path or getenv("OUTPUT_PATH") or suggested_name
    with open(out_path, "wb") as f:
        f.write(png_bytes)
    return out_path


if __name__ == "__main__":
    # CLI entry: use env vars with sensible defaults matching previous script behavior
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    default_from = "2025-08-25T11:00:00Z"
    default_to = "2025-08-25T13:00:00Z"
    default_agent = "skillcompanion_interrupted"
    from_env = getenv("FROM_TIMESTAMP") or default_from
    to_env = getenv("TO_TIMESTAMP") or default_to
    out_env = getenv("OUTPUT_PATH")
    try:
        saved = generate_skillcompanion_png(
            from_env, to_env, agent_name=default_agent, output_path=out_env
        )
        print(f"Saved bar chart to {saved}")
    except Exception as e:
        # Print error and exit non-zero

        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
