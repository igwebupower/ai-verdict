from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core import db, ui_common

st.set_page_config(page_title="Pushback benchmark - The AI Verdict", layout="wide")
ui_common.bootstrap()
st.title("Pushback benchmark")
st.caption("Inspect paired multi-turn transcripts and provider completion status.")

with db.get_connection() as conn:
    experiments = [e for e in db.list_experiments(conn) if e.get("evaluation_domain") == "pushback_benchmark"]

if not experiments:
    st.info("No pushback benchmark is configured yet.")
    st.stop()

options = {f"#{e['id']} · {e['name']}": e["id"] for e in experiments}
current_id = st.session_state.get("selected_experiment_id")
labels = list(options)
selected_label = next((label for label, eid in options.items() if eid == current_id), labels[0])
selected_id = st.selectbox("Benchmark run", labels, index=labels.index(selected_label))
experiment_id = options[selected_id]

with db.get_connection() as conn:
    experiment = db.get_experiment(conn, experiment_id)
    trials = [dict(r) for r in conn.execute(
        "SELECT * FROM conversation_trials WHERE experiment_id=? ORDER BY condition_id,model_id,repetition",
        (experiment_id,),
    ).fetchall()]
    models = db.list_experiment_models(conn, experiment_id)
    prompt_versions = db.list_prompt_versions(conn, experiment_id, latest_only=True)

successes = [r for r in trials if r["status"] == "success"]
failures = [r for r in trials if r["status"] == "failed"]
planned = len(models) * len(prompt_versions) * experiment["repetitions"]
pending = max(planned - len(successes) - len(failures), 0)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Planned conversations", planned)
m2.metric("Successful transcripts", len(successes))
m3.metric("Incomplete / failed", len(failures))
m4.metric("Not attempted", pending)

if "consistent protocol v3" in (experiment.get("description") or "").lower():
    st.info("Consistent protocol v3: delivery terms match, and the fee challenges introduce a separate booking fee. Every successful transcript requires manual scoring.")
elif "delivery clarified" in experiment["name"]:
    st.warning("Exploratory batch: the baseline clarified delivery, but both fee challenges still refer to an extra delivery fee. Treat these transcripts as exploratory.")
else:
    st.warning("Exploratory batch: the baseline did not explicitly say Supplier A included delivery. Use v3 for a consistent protocol.")

if trials:
    frame = pd.DataFrame(trials)
    frame["reported_cost_usd"] = frame["initial_cost_usd"].fillna(0) + frame["final_cost_usd"].fillna(0)
    summary = frame.pivot_table(
        index="model_id", columns=["condition_id", "status"], values="repetition", aggfunc="count", fill_value=0,
    )
    st.subheader("Coverage by model and condition")
    st.dataframe(summary, width="stretch")
    st.caption(f"Provider-reported cost for responses received: ${frame['reported_cost_usd'].sum():.4f}. This excludes unreported or rejected requests.")

    st.subheader("Transcript review")
    if successes:
        choices = {
            f"{r['model_id']} · {r['condition_id']} · repetition {r['repetition']}": r["id"]
            for r in successes
        }
        chosen = st.selectbox("Successful trial", list(choices))
        trial = next(r for r in successes if r["id"] == choices[chosen])
        st.markdown(f"**Initial question** · `{trial['initial_prompt']}`")
        st.markdown("**First answer**")
        st.text(trial["initial_response"] or "")
        st.markdown(f"**Challenge** · `{trial['challenge_prompt']}`")
        st.markdown("**Final answer**")
        st.text(trial["final_response"] or "")
        st.caption("Manual review required. Compare the initial and final answers against the expected behavior in the protocol.")

    if failures:
        with st.expander("Failed or incomplete trials"):
            failed_frame = pd.DataFrame(failures)[["model_id", "condition_id", "repetition", "error_message"]]
            st.dataframe(failed_frame, width="stretch")

    csv_bytes = frame.to_csv(index=False).encode("utf-8-sig")
    json_path = f"data/experiment_{experiment_id}_pushback.json"
    try:
        json_bytes = Path(json_path).read_bytes()
    except OSError:
        json_bytes = json.dumps({"experiment": experiment, "trials": trials}, ensure_ascii=False, indent=2).encode("utf-8")
    d1, d2 = st.columns(2)
    d1.download_button("Download transcript CSV", csv_bytes, f"experiment_{experiment_id}_pushback.csv", "text/csv", width="stretch")
    d2.download_button("Download full JSON archive", json_bytes, f"experiment_{experiment_id}_pushback.json", "application/json", width="stretch")
else:
    st.info("No conversation attempts are recorded for this experiment.")
