from datetime import datetime, timedelta, timezone
from os import environ, getenv
import json
import requests
from langfuse import Langfuse
from pandas import concat, DataFrame
from pandas import to_datetime, read_parquet
import matplotlib
import matplotlib.pyplot as plt

def langfuse_trace_call(public_key, secret_key, lf_url, page,name,fromTimestamp,toTimestamp):

        response = requests.get(
            f"{lf_url}/api/public/traces",
            auth=(
                f"{public_key}",
                f"{secret_key}",
            ),
            params={"page": f"{page}",
                    "name": f"{name}",
                    "fromTimestamp": f"{fromTimestamp}",
                    "toTimestamp": f"{toTimestamp}"}
        )
        try:
            response.raise_for_status()
            return response
        except requests.HTTPError as err:
            raise Exception(f"HTTP error occurred: {err}")
        except Exception as err:
            raise Exception(f"Other error occurred: {err}")

def get_users(tenant):
    token = tenant['token']
    base_url = tenant['owuiurl']
    headers = {
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
        }
    response = requests.get(f"{base_url}/api/v1/users/", headers=headers)

    return response.json()

def get_LF_traces(public_key, secret_key, lfurl, name_agent,fromTimestamp,toTimestamp,start=0, end=-1,):
    
        n = 1
        l = 1

        traces = []
        while l > 0:
            trace = langfuse_trace_call(public_key, secret_key, lfurl, page=n,name=name_agent,fromTimestamp=fromTimestamp,toTimestamp=toTimestamp)
            traces = traces + trace.json()["data"]
            n = n + 1
            l = len(trace.json()["data"])

        return traces

tenants = [
    {
        "user": "demo",
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjQwNDA5NTRmLWUzN2UtNGFkYy04N2VlLWFhZTUwMmNjY2FmZiJ9.62OPYfEWJjURaDCmpvT21GRYtmRByHm-OPSffDjgJno",
        "lfsk": "sk-lf-9be92eb2-0000-434c-8c1b-efecd2d50ff4",
        "lfpk": "pk-lf-7de86adb-d32f-403e-8ce0-c4998701efea",
        "owuiurl": 'https://demo.roosi.ai',
        "lfurl":  "https://langfuse.roosi.ai",
        "name" : "skillcompanion_interrupted",
        "fromTimestamp":"2025-08-25T11:00:00Z",
        "toTimestamp":"2025-08-25T13:00:00Z"
    },
    ]


trace_dataframe = DataFrame()
extracted = {}
for tenant in tenants:
    if tenant['user'] not in ["akdb"]:
        print(f"Extracting {tenant['user']} traces")
        # print(datetime.now())
        langfuse = Langfuse(public_key= tenant['lfpk'], secret_key= tenant['lfsk'], host=tenant['lfurl'])

        # Allow overrides via environment variables passed from Taskfile
        from_env = getenv("FROM_TIMESTAMP")
        to_env = getenv("TO_TIMESTAMP")
        name_env = getenv("AGENT_NAME")

        name = name_env if name_env else tenant['name']
        from_ts = from_env if from_env else tenant['fromTimestamp']
        to_ts = to_env if to_env else tenant['toTimestamp']

        traces = get_LF_traces(
            tenant['lfpk'],
            tenant['lfsk'],
            tenant['lfurl'],
            name_agent=name,
            fromTimestamp=from_ts,
            toTimestamp=to_ts
        )
        extracted[tenant['user']] = traces

rows = []
for entry in extracted[tenants[0]['user']]:
    try:
        row = {
            'id': entry.get('id'),
            'timestamp': entry.get('timestamp'),
            'userId': entry.get('userId'),
            'sessionId': entry.get('sessionId'),
            'username': entry.get('input', {}).get('varables', {}).get('{{USER_NAME}}'),
            'output': entry.get('output'),
            'category': entry['output'].get('category')
        }
        rows.append(row)
    except AttributeError as e:
         print(f"Exception: {e}")

df = DataFrame(rows)

beginner_count = df['category'].str.contains(r'Beginner', case=False, na=False).sum()
advanced_count = df['category'].str.contains(r'Advanced', case=False, na=False).sum()
expert_count = df['category'].str.contains(r'Expert', case=False, na=False).sum()

# Create bar chart for skill categories
try:
    
    matplotlib.use("Agg")  # Use non-interactive backend suitable for headless environments
    

    labels = ["Beginner", "Advanced", "Expert"]
    values = [int(beginner_count), int(advanced_count), int(expert_count)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=["#4CAF50", "#2196F3", "#9C27B0"])
    ax.set_title("Skill Check Ergebnisse", pad=16)
    ax.set_ylabel("Anzahl")
    upper = max(values) if max(values) > 0 else 1
    ax.set_ylim(0, upper * 1.3)  # add headroom above bars
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # integer-only ticks
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)  # add grid on y-axis
    ax.bar_label(bars, labels=[str(v) for v in values], padding=3)

    fig.tight_layout()
    output_path = "skill_levels_bar_chart.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved bar chart to {output_path}")
except ImportError:
    print("matplotlib is not installed. Install with: 'uv add matplotlib' or 'pip install matplotlib' to generate the chart.")
except Exception as e:
    print(f"Chart generation failed: {e}")

