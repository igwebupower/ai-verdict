"""The AI Verdict - LLM Experiment Lab. Home page: list, duplicate and open
experiments. See pages/ for the rest of the workflow."""
from __future__ import annotations

import streamlit as st

from core import db, runner, seed, sentencing_seed, ui_common

st.set_page_config(page_title="The AI Verdict - Experiments", layout="wide")
ui_common.bootstrap()

st.title("The AI Verdict - LLM Experiment Lab")
st.caption("Independently test AI models on real-world tasks and publish evidence-based findings.")


def _duplicate_experiment(conn, experiment_id: int) -> int:
    src = db.get_experiment(conn, experiment_id)
    new_id = db.create_experiment(
        conn, name=f"{src['name']} (copy)", hypothesis=src["hypothesis"], description=src["description"],
        repetitions=src["repetitions"], randomize_order=bool(src["randomize_order"]),
        temperature=src["temperature"], max_tokens=src["max_tokens"], top_p=src["top_p"],
        frequency_penalty=src["frequency_penalty"], presence_penalty=src["presence_penalty"], seed=src["seed"],
        evaluator_model_id=src["evaluator_model_id"], evaluation_domain=src["evaluation_domain"],
        source_experiment_id=experiment_id,
    )
    for pv in db.list_prompt_versions(conn, experiment_id, latest_only=True):
        db.create_prompt_version(conn, new_id, pv["variant_name"], pv["prompt_text"], notes=pv["notes"],
                                  is_role_reversed=bool(pv["is_role_reversed"]))
    for m in db.list_experiment_models(conn, experiment_id):
        db.add_experiment_model(conn, new_id, m["model_id"], m["provider"], m["display_name"])
    return new_id

if not ui_common.api_key_configured():
    st.warning(
        "OPENROUTER_API_KEY is not set. You can still browse and create experiments, but running them "
        "requires an API key in `.env` - see `.env.example` and the **Settings** page."
    )

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    if st.button("Load gender-bias sample experiment", use_container_width=True,
                  help='Seeds "Do LLMs Assume Nurses Are Women?" with its baseline, role-reversed and '
                       'alternative-profession prompt variants.'):
        with db.get_connection() as conn:
            exp_id = seed.create_builtin_gender_experiment(conn)
        st.session_state["selected_experiment_id"] = exp_id
        st.success(f"Sample experiment ready (id #{exp_id}). Open **Create Experiment** to pick models, "
                   f"or **Run Experiment** if models are already selected.")
with col_b:
    if st.button("Load sentencing-bias sample experiment", use_container_width=True,
                  help='Seeds "The AI judge gave him a longer sentence." with its baseline, name-swap and '
                       'three dialect prompt variants.'):
        with db.get_connection() as conn:
            exp_id = sentencing_seed.create_builtin_sentencing_experiment(conn)
        st.session_state["selected_experiment_id"] = exp_id
        st.success(f"Sample experiment ready (id #{exp_id}). Open **Create Experiment** to pick models, "
                   f"or **Run Experiment** if models are already selected.")
with col_c:
    if st.button("Load name/nationality sample experiment", use_container_width=True,
                  help='Seeds "Would he get the same sentence with a different name?" - same case and '
                       'statement as the sentencing-bias sample, only the defendant\'s name changes across '
                       'English, Afghan, Chinese, and Nigerian variants.'):
        with db.get_connection() as conn:
            exp_id = sentencing_seed.create_builtin_name_nationality_experiment(conn)
        st.session_state["selected_experiment_id"] = exp_id
        st.success(f"Sample experiment ready (id #{exp_id}). Open **Create Experiment** to pick models, "
                   f"or **Run Experiment** if models are already selected.")
with col_d:
    st.page_link("pages/1_Create_Experiment.py", label="Create a new experiment", icon=":material/add:")

st.divider()

with db.get_connection() as conn:
    experiments = db.list_experiments(conn)

if not experiments:
    st.info("No experiments yet.")
else:
    st.subheader("Experiments")
    for exp in experiments:
        with db.get_connection() as conn:
            runs = db.list_runs(conn, exp["id"])
            prompt_versions = db.list_prompt_versions(conn, exp["id"], latest_only=True)
            models = db.list_experiment_models(conn, exp["id"])
            if exp.get("evaluation_domain") == "pushback_benchmark":
                trial_counts = {r["status"]: r["n"] for r in conn.execute(
                    "SELECT status, COUNT(*) n FROM conversation_trials WHERE experiment_id = ? GROUP BY status",
                    (exp["id"],),
                )}
            else:
                trial_counts = None
        if trial_counts is not None:
            success = trial_counts.get("success", 0)
            failed = trial_counts.get("failed", 0)
            planned = len(prompt_versions) * len(models) * exp["repetitions"]
            pending = max(planned - success - failed, 0)
        else:
            success = sum(1 for r in runs if r["status"] == "success")
            failed = sum(1 for r in runs if r["status"] == "failed")
            pending = sum(1 for r in runs if r["status"] in ("pending", "running"))
            planned = runner.estimate_call_count(max(len(prompt_versions), 1), max(len(models), 1), exp["repetitions"])

        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"**#{exp['id']} - {exp['name']}**")
                st.caption(exp["hypothesis"] or "_no hypothesis recorded_")
            with c2:
                st.write(f"Status: `{exp['status']}`")
                st.write(f"Runs: {success} ok / {failed} failed / {pending} pending (planned ~{planned})")
            with c3:
                b1, b2, b3 = st.columns(3)
                if b1.button("Open", key=f"open_{exp['id']}"):
                    st.session_state["selected_experiment_id"] = exp["id"]
                    st.switch_page(
                        "pages/9_Pushback_Benchmark.py"
                        if exp.get("evaluation_domain") == "pushback_benchmark"
                        else "pages/2_Run_Experiment.py"
                    )
                if b2.button("Duplicate", key=f"dup_{exp['id']}"):
                    with db.get_connection() as conn:
                        new_id = _duplicate_experiment(conn, exp["id"])
                    st.session_state["selected_experiment_id"] = new_id
                    st.rerun()
                if b3.button("Dashboard", key=f"dash_{exp['id']}"):
                    st.session_state["selected_experiment_id"] = exp["id"]
                    st.switch_page("pages/3_Results_Dashboard.py")
