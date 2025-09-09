from os import getenv

import matplotlib
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
from pandas import DataFrame

# Load .env so subprocess can pick up environment configuration
load_dotenv()


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
    )
    try:
        response.raise_for_status()
        return response
    except requests.HTTPError as err:
        raise Exception(f"HTTP error occurred: {err}")
    except Exception as err:
        raise Exception(f"Other error occurred: {err}")


def get_LF_traces(
    public_key: str,
    secret_key: str,
    lfurl: str,
    name_agent: str,
    fromTimestamp: str,
    toTimestamp: str,
    start: int = 0,
    end: int = -1,
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


def generate_skillcompanion_png(
    from_timestamp: str,
    to_timestamp: str,
    agent_name: str = "skillcompanion_interrupted",
    output_path: str | None = None,
) -> str:
    """
    Generate the Skill Companion bar chart PNG for the given time range.

    Args:
        from_timestamp: ISO8601 timestamp (e.g. 2025-08-25T00:00:00Z)
        to_timestamp: ISO8601 timestamp (e.g. 2025-08-25T23:59:59Z)
        agent_name: Langfuse trace name to filter on
        output_path: If provided, save PNG to this path; otherwise use OUTPUT_PATH env var or default file name

    Returns:
        The file system path where the PNG was written.
    """
    # Read configuration from environment (.env)
    lf_public_key = getenv("LANGFUSE_PUBLIC_KEY")
    lf_secret_key = getenv("LANGFUSE_SECRET_KEY")
    lf_url = getenv("LANGFUSE_HOST")
    # Optional: present for completeness if needed later

    if not lf_public_key or not lf_secret_key or not lf_url:
        raise RuntimeError(
            "Missing required env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST must be set."
        )

    traces = get_LF_traces(
        lf_public_key,
        lf_secret_key,
        lf_url,
        name_agent=agent_name,
        fromTimestamp=from_timestamp,
        toTimestamp=to_timestamp,
    )

    # Build DataFrame rows
    rows: list[dict] = []
    for entry in traces:
        try:
            row = {
                "id": entry.get("id"),
                "timestamp": entry.get("timestamp"),
                "userId": entry.get("userId"),
                "sessionId": entry.get("sessionId"),
                "username": entry.get("input", {}).get("variables", {}).get("{{USER_NAME}}"),
                "output": entry.get("output"),
                "category": (entry.get("output") or {}).get("category")
                if isinstance(entry.get("output"), dict)
                else None,
            }
            rows.append(row)
        except Exception:
            # Ignore malformed entries
            pass

    df = DataFrame(rows)

    # Counts
    if not df.empty and "category" in df.columns:
        beginner_count = df["category"].str.contains(r"Beginner", case=False, na=False).sum()
        advanced_count = df["category"].str.contains(r"Advanced", case=False, na=False).sum()
        expert_count = df["category"].str.contains(r"Expert", case=False, na=False).sum()
    else:
        beginner_count = 0
        advanced_count = 0
        expert_count = 0

    # Plot
    try:
        matplotlib.use("Agg")  # Non-interactive backend
        labels = ["Beginner", "Advanced", "Expert"]
        values = [int(beginner_count), int(advanced_count), int(expert_count)]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.bar(labels, values, color=["#4CAF50", "#2196F3", "#9C27B0"])
        ax.set_title("Skill Check Ergebnisse", pad=16)
        ax.set_ylabel("Anzahl")
        upper = max(values) if max(values) > 0 else 1
        ax.set_ylim(0, upper * 1.3)
        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.bar_label(bars, labels=[str(v) for v in values], padding=3)
        fig.tight_layout()
        # Resolve output path
        out_path = output_path or getenv("OUTPUT_PATH") or "skill_levels_bar_chart.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return out_path
    except ImportError:
        raise RuntimeError(
            "matplotlib is not installed. Install with: 'uv add matplotlib' or 'pip install matplotlib'."
        )
    except Exception as e:
        raise RuntimeError(f"Chart generation failed: {e}")


if __name__ == "__main__":
    # CLI entry: use env vars with sensible defaults matching previous script behavior
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
        import sys

        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
