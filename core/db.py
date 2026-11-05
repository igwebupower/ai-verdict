"""SQLite persistence layer for AI Verdict.

Every experiment, prompt version, run, evaluation and human review is stored
here. Connections are short-lived (opened per call) because Streamlit reruns
the whole script on every interaction - this avoids holding a long-lived
connection across reruns while still being safe for a local, single-user app.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ai_verdict.db"


def get_db_path() -> Path:
    raw = os.environ.get("AI_VERDICT_DB_PATH")
    path = Path(raw) if raw else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path else get_db_path()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    hypothesis TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    repetitions INTEGER NOT NULL DEFAULT 1,
    randomize_order INTEGER NOT NULL DEFAULT 0,
    temperature REAL,
    max_tokens INTEGER,
    top_p REAL,
    frequency_penalty REAL,
    presence_penalty REAL,
    seed INTEGER,
    extra_params_json TEXT NOT NULL DEFAULT '{}',
    reveal_hypothesis_to_model INTEGER NOT NULL DEFAULT 0,
    evaluator_model_id TEXT,
    evaluation_domain TEXT NOT NULL DEFAULT 'gender_healthcare',
    status TEXT NOT NULL DEFAULT 'draft',
    source_experiment_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    variant_name TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    is_role_reversed INTEGER NOT NULL DEFAULT 0,
    is_locked INTEGER NOT NULL DEFAULT 0,
    superseded_by INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    provider TEXT,
    display_name TEXT,
    supported_params_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE(experiment_id, model_id)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    prompt_version_id INTEGER NOT NULL REFERENCES prompt_versions(id),
    model_id TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    sequence_number INTEGER NOT NULL,
    execution_order INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    request_params_json TEXT NOT NULL DEFAULT '{}',
    unsupported_params_json TEXT NOT NULL DEFAULT '[]',
    raw_response TEXT,
    response_id TEXT,
    response_model TEXT,
    finish_reason TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    http_status INTEGER,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    retried_from_run_id INTEGER REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    evaluator_model_id TEXT NOT NULL,
    evaluator_prompt_version TEXT NOT NULL,
    raw_evaluator_response TEXT,
    parsed_json TEXT,
    is_valid INTEGER NOT NULL DEFAULT 0,
    validation_errors TEXT,
    doctor_gender_classification TEXT,
    doctor_assignment_type TEXT,
    doctor_quote TEXT,
    doctor_reasoning TEXT,
    nurse_gender_classification TEXT,
    nurse_assignment_type TEXT,
    nurse_quote TEXT,
    nurse_reasoning TEXT,
    uncertain INTEGER NOT NULL DEFAULT 0,
    uncertain_reason TEXT,
    sentence_type TEXT,
    sentence_custodial_months REAL,
    sentence_quote TEXT,
    sentence_reasoning TEXT,
    sentence_no_recommendation INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL,
    attempt_number INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    evaluation_id INTEGER REFERENCES evaluations(id),
    reviewer_name TEXT NOT NULL DEFAULT '',
    final_doctor_classification TEXT,
    final_doctor_assignment_type TEXT,
    final_nurse_classification TEXT,
    final_nurse_assignment_type TEXT,
    final_sentence_type TEXT,
    final_sentence_custodial_months REAL,
    evidence_insufficient INTEGER NOT NULL DEFAULT 0,
    reviewer_note TEXT NOT NULL DEFAULT '',
    is_latest INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_catalog_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL UNIQUE,
    name TEXT,
    context_length INTEGER,
    pricing_prompt TEXT,
    pricing_completion TEXT,
    supported_parameters_json TEXT NOT NULL DEFAULT '[]',
    raw_json TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    condition_id TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    initial_prompt TEXT NOT NULL,
    initial_response TEXT,
    challenge_prompt TEXT NOT NULL,
    final_response TEXT,
    initial_response_id TEXT,
    final_response_id TEXT,
    initial_response_model TEXT,
    final_response_model TEXT,
    initial_cost_usd REAL,
    final_cost_usd REAL,
    initial_latency_ms INTEGER,
    final_latency_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(experiment_id, model_id, condition_id, repetition)
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_human_reviews_run ON human_reviews(run_id, is_latest);
"""


def init_db(db_path: str | Path | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """CREATE TABLE IF NOT EXISTS never alters an already-existing table, so
    a column added to SCHEMA after a database was first created needs an
    explicit, idempotent ALTER TABLE here too - otherwise anyone with an
    existing database silently never gets the new column."""
    run_cols = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    if "retried_from_run_id" not in run_cols:
        conn.execute("ALTER TABLE runs ADD COLUMN retried_from_run_id INTEGER REFERENCES runs(id)")

    experiment_cols = {row["name"] for row in conn.execute("PRAGMA table_info(experiments)")}
    if "evaluation_domain" not in experiment_cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN evaluation_domain TEXT NOT NULL DEFAULT 'gender_healthcare'")

    eval_cols = {row["name"] for row in conn.execute("PRAGMA table_info(evaluations)")}
    for col_sql in (
        "sentence_type TEXT",
        "sentence_custodial_months REAL",
        "sentence_quote TEXT",
        "sentence_reasoning TEXT",
        "sentence_no_recommendation INTEGER NOT NULL DEFAULT 0",
        "cost_usd REAL",
    ):
        col_name = col_sql.split()[0]
        if col_name not in eval_cols:
            conn.execute(f"ALTER TABLE evaluations ADD COLUMN {col_sql}")

    review_cols = {row["name"] for row in conn.execute("PRAGMA table_info(human_reviews)")}
    for col_sql in ("final_sentence_type TEXT", "final_sentence_custodial_months REAL"):
        col_name = col_sql.split()[0]
        if col_name not in review_cols:
            conn.execute(f"ALTER TABLE human_reviews ADD COLUMN {col_sql}")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def create_experiment(
    conn: sqlite3.Connection,
    name: str,
    hypothesis: str = "",
    description: str = "",
    repetitions: int = 1,
    randomize_order: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    extra_params: dict | None = None,
    reveal_hypothesis_to_model: bool = False,
    evaluator_model_id: str | None = None,
    evaluation_domain: str = "gender_healthcare",
    source_experiment_id: int | None = None,
) -> int:
    ts = now_iso()
    cur = conn.execute(
        """
        INSERT INTO experiments
            (name, hypothesis, description, repetitions, randomize_order,
             temperature, max_tokens, top_p, frequency_penalty, presence_penalty,
             seed, extra_params_json, reveal_hypothesis_to_model, evaluator_model_id, evaluation_domain,
             status, source_experiment_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?, ?)
        """,
        (
            name, hypothesis, description, repetitions, int(randomize_order),
            temperature, max_tokens, top_p, frequency_penalty, presence_penalty,
            seed, json.dumps(extra_params or {}), int(reveal_hypothesis_to_model), evaluator_model_id,
            evaluation_domain, source_experiment_id, ts, ts,
        ),
    )
    return cur.lastrowid


def get_experiment(conn: sqlite3.Connection, experiment_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    return row_to_dict(row)


def list_experiments(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM experiments ORDER BY updated_at DESC").fetchall()
    return rows_to_dicts(rows)


def update_experiment_status(conn: sqlite3.Connection, experiment_id: int, status: str) -> None:
    conn.execute(
        "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
        (status, now_iso(), experiment_id),
    )


def update_experiment_fields(conn: sqlite3.Connection, experiment_id: int, **fields: Any) -> None:
    if not fields:
        return
    if "extra_params" in fields:
        fields["extra_params_json"] = json.dumps(fields.pop("extra_params"))
    cols = ", ".join(f"{k} = ?" for k in fields)
    params = list(fields.values()) + [now_iso(), experiment_id]
    conn.execute(f"UPDATE experiments SET {cols}, updated_at = ? WHERE id = ?", params)


def has_any_runs(conn: sqlite3.Connection, experiment_id: int) -> bool:
    row = conn.execute("SELECT COUNT(*) c FROM runs WHERE experiment_id = ?", (experiment_id,)).fetchone()
    return row["c"] > 0


# ---------------------------------------------------------------------------
# Prompt versions
# ---------------------------------------------------------------------------

def create_prompt_version(
    conn: sqlite3.Connection,
    experiment_id: int,
    variant_name: str,
    prompt_text: str,
    notes: str = "",
    is_role_reversed: bool = False,
    version_number: int | None = None,
) -> int:
    if version_number is None:
        row = conn.execute(
            "SELECT COALESCE(MAX(version_number), 0) v FROM prompt_versions "
            "WHERE experiment_id = ? AND variant_name = ?",
            (experiment_id, variant_name),
        ).fetchone()
        version_number = row["v"] + 1
    cur = conn.execute(
        """
        INSERT INTO prompt_versions
            (experiment_id, variant_name, version_number, prompt_text, notes,
             is_role_reversed, is_locked, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (experiment_id, variant_name, version_number, prompt_text, notes,
         int(is_role_reversed), now_iso()),
    )
    return cur.lastrowid


def list_prompt_versions(conn: sqlite3.Connection, experiment_id: int, latest_only: bool = False) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM prompt_versions WHERE experiment_id = ? ORDER BY variant_name, version_number",
        (experiment_id,),
    ).fetchall()
    items = rows_to_dicts(rows)
    if not latest_only:
        return items
    latest: dict[str, dict[str, Any]] = {}
    for item in items:
        latest[item["variant_name"]] = item
    return list(latest.values())


def get_prompt_version(conn: sqlite3.Connection, prompt_version_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM prompt_versions WHERE id = ?", (prompt_version_id,)).fetchone()
    return row_to_dict(row)


def lock_prompt_version(conn: sqlite3.Connection, prompt_version_id: int) -> None:
    conn.execute("UPDATE prompt_versions SET is_locked = 1 WHERE id = ?", (prompt_version_id,))


def edit_prompt_version(
    conn: sqlite3.Connection,
    prompt_version_id: int,
    new_prompt_text: str,
    notes: str = "",
) -> int:
    """Create a new, unlocked version. If runs already reference the old
    version it stays immutable and is marked superseded - this is how prompt
    edits after runs have started are handled without ever rewriting history.
    """
    old = get_prompt_version(conn, prompt_version_id)
    if old is None:
        raise ValueError(f"prompt version {prompt_version_id} not found")
    new_id = create_prompt_version(
        conn, old["experiment_id"], old["variant_name"], new_prompt_text,
        notes=notes, is_role_reversed=bool(old["is_role_reversed"]),
    )
    conn.execute("UPDATE prompt_versions SET superseded_by = ? WHERE id = ?", (new_id, prompt_version_id))
    return new_id


# ---------------------------------------------------------------------------
# Experiment models (model selection per experiment)
# ---------------------------------------------------------------------------

def add_experiment_model(
    conn: sqlite3.Connection,
    experiment_id: int,
    model_id: str,
    provider: str | None = None,
    display_name: str | None = None,
    supported_params: list[str] | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO experiment_models (experiment_id, model_id, provider, display_name, supported_params_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment_id, model_id) DO UPDATE SET
            provider = excluded.provider,
            display_name = excluded.display_name,
            supported_params_json = excluded.supported_params_json
        """,
        (experiment_id, model_id, provider, display_name, json.dumps(supported_params or []), now_iso()),
    )
    return cur.lastrowid


def remove_experiment_model(conn: sqlite3.Connection, experiment_id: int, model_id: str) -> None:
    conn.execute("DELETE FROM experiment_models WHERE experiment_id = ? AND model_id = ?", (experiment_id, model_id))


def list_experiment_models(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM experiment_models WHERE experiment_id = ? ORDER BY model_id", (experiment_id,)
    ).fetchall()
    return rows_to_dicts(rows)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def create_run(
    conn: sqlite3.Connection,
    experiment_id: int,
    prompt_version_id: int,
    model_id: str,
    run_number: int,
    sequence_number: int,
    request_params: dict,
    unsupported_params: list[str] | None = None,
    retried_from_run_id: int | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO runs
            (experiment_id, prompt_version_id, model_id, run_number, sequence_number,
             status, request_params_json, unsupported_params_json, retried_from_run_id, created_at)
        VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
        """,
        (experiment_id, prompt_version_id, model_id, run_number, sequence_number,
         json.dumps(request_params), json.dumps(unsupported_params or []), retried_from_run_id, now_iso()),
    )
    return cur.lastrowid


def get_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return row_to_dict(row)


def list_runs(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runs WHERE experiment_id = ? ORDER BY sequence_number", (experiment_id,)
    ).fetchall()
    return rows_to_dicts(rows)


def get_pending_runs(conn: sqlite3.Connection, experiment_id: int, execution_order: bool = False) -> list[dict[str, Any]]:
    order_col = "execution_order" if execution_order else "sequence_number"
    rows = conn.execute(
        f"SELECT * FROM runs WHERE experiment_id = ? AND status IN ('pending', 'running') "
        f"ORDER BY {order_col} IS NULL, {order_col}, sequence_number",
        (experiment_id,),
    ).fetchall()
    return rows_to_dicts(rows)


def set_run_execution_order(conn: sqlite3.Connection, run_id: int, execution_order: int) -> None:
    conn.execute("UPDATE runs SET execution_order = ? WHERE id = ?", (execution_order, run_id))


def mark_run_running(conn: sqlite3.Connection, run_id: int) -> None:
    conn.execute(
        "UPDATE runs SET status = 'running', started_at = ? WHERE id = ?", (now_iso(), run_id)
    )


def mark_run_success(
    conn: sqlite3.Connection,
    run_id: int,
    raw_response: str,
    response_id: str | None,
    response_model: str | None,
    finish_reason: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cost_usd: float | None,
    latency_ms: int,
    retry_count: int,
) -> None:
    conn.execute(
        """
        UPDATE runs SET
            status = 'success', raw_response = ?, response_id = ?, response_model = ?,
            finish_reason = ?, prompt_tokens = ?, completion_tokens = ?, total_tokens = ?,
            cost_usd = ?, latency_ms = ?, retry_count = ?, completed_at = ?
        WHERE id = ?
        """,
        (raw_response, response_id, response_model, finish_reason, prompt_tokens,
         completion_tokens, total_tokens, cost_usd, latency_ms, retry_count, now_iso(), run_id),
    )


def mark_run_failed(
    conn: sqlite3.Connection,
    run_id: int,
    error_message: str,
    http_status: int | None,
    latency_ms: int,
    retry_count: int,
) -> None:
    conn.execute(
        """
        UPDATE runs SET
            status = 'failed', error_message = ?, http_status = ?, latency_ms = ?,
            retry_count = ?, completed_at = ?
        WHERE id = ?
        """,
        (error_message, http_status, latency_ms, retry_count, now_iso(), run_id),
    )


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------

def create_evaluation(conn: sqlite3.Connection, run_id: int, evaluator_model_id: str,
                       evaluator_prompt_version: str, attempt_number: int = 1) -> int:
    cur = conn.execute(
        """
        INSERT INTO evaluations (run_id, evaluator_model_id, evaluator_prompt_version, attempt_number, status, created_at)
        VALUES (?, ?, ?, ?, 'pending', ?)
        """,
        (run_id, evaluator_model_id, evaluator_prompt_version, attempt_number, now_iso()),
    )
    return cur.lastrowid


def update_evaluation_result(
    conn: sqlite3.Connection,
    evaluation_id: int,
    raw_evaluator_response: str,
    is_valid: bool,
    validation_errors: str | None,
    parsed: dict | None,
    cost_usd: float | None = None,
) -> None:
    doctor = (parsed or {}).get("doctor", {})
    nurse = (parsed or {}).get("nurse", {})
    status = "valid" if is_valid else "invalid_needs_review"
    conn.execute(
        """
        UPDATE evaluations SET
            raw_evaluator_response = ?, parsed_json = ?, is_valid = ?, validation_errors = ?,
            doctor_gender_classification = ?, doctor_assignment_type = ?, doctor_quote = ?, doctor_reasoning = ?,
            nurse_gender_classification = ?, nurse_assignment_type = ?, nurse_quote = ?, nurse_reasoning = ?,
            uncertain = ?, uncertain_reason = ?, cost_usd = ?, status = ?
        WHERE id = ?
        """,
        (
            raw_evaluator_response, json.dumps(parsed) if parsed else None, int(is_valid), validation_errors,
            doctor.get("gender_classification"), doctor.get("assignment_type"), doctor.get("quote"), doctor.get("reasoning"),
            nurse.get("gender_classification"), nurse.get("assignment_type"), nurse.get("quote"), nurse.get("reasoning"),
            int((parsed or {}).get("uncertain", False)), (parsed or {}).get("uncertain_reason"),
            cost_usd, status, evaluation_id,
        ),
    )


def update_sentencing_evaluation_result(
    conn: sqlite3.Connection,
    evaluation_id: int,
    raw_evaluator_response: str,
    is_valid: bool,
    validation_errors: str | None,
    parsed: dict | None,
    cost_usd: float | None = None,
) -> None:
    status = "valid" if is_valid else "invalid_needs_review"
    p = parsed or {}
    conn.execute(
        """
        UPDATE evaluations SET
            raw_evaluator_response = ?, parsed_json = ?, is_valid = ?, validation_errors = ?,
            sentence_type = ?, sentence_custodial_months = ?, sentence_quote = ?, sentence_reasoning = ?,
            sentence_no_recommendation = ?, cost_usd = ?, status = ?
        WHERE id = ?
        """,
        (
            raw_evaluator_response, json.dumps(parsed) if parsed else None, int(is_valid), validation_errors,
            p.get("sentence_type"), p.get("custodial_months"), p.get("quote"), p.get("reasoning"),
            int(p.get("sentence_type") == "no_recommendation"), cost_usd, status, evaluation_id,
        ),
    )


def mark_evaluation_error(conn: sqlite3.Connection, evaluation_id: int, error_message: str) -> None:
    conn.execute(
        "UPDATE evaluations SET status = 'error', validation_errors = ? WHERE id = ?",
        (error_message, evaluation_id),
    )


def get_latest_evaluation_for_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM evaluations WHERE run_id = ? ORDER BY attempt_number DESC, id DESC LIMIT 1", (run_id,)
    ).fetchone()
    return row_to_dict(row)


def list_evaluations_for_experiment(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    """Ordered so the first row seen per run_id is the latest evaluation
    attempt. attempt_number alone is not a reliable tiebreaker - nothing in
    this codebase currently increments it on retry (every retry is recorded
    as attempt_number=1), so id DESC is the real recency signal; without it,
    SQLite's tie order is unspecified and callers like
    analysis.build_experiment_rows (which keeps the first row per run_id via
    setdefault) could silently pick a stale, superseded evaluation."""
    rows = conn.execute(
        """
        SELECT e.* FROM evaluations e
        JOIN runs r ON r.id = e.run_id
        WHERE r.experiment_id = ?
        ORDER BY e.run_id, e.attempt_number DESC, e.id DESC
        """,
        (experiment_id,),
    ).fetchall()
    return rows_to_dicts(rows)


def get_latest_evaluations_by_run(conn: sqlite3.Connection, experiment_id: int) -> dict[int, dict[str, Any]]:
    """One evaluation per run_id - the most recent attempt - built from a
    single JOINed query instead of one query per run (get_latest_evaluation_
    for_run). Use this whenever you need the latest evaluation for more than
    one run in an experiment; analysis.py, runner.py, and the Human Review
    page all need exactly this, so it lives here once instead of three times."""
    latest: dict[int, dict[str, Any]] = {}
    for ev in list_evaluations_for_experiment(conn, experiment_id):
        latest.setdefault(ev["run_id"], ev)
    return latest


# ---------------------------------------------------------------------------
# Human reviews
# ---------------------------------------------------------------------------

def create_human_review(
    conn: sqlite3.Connection,
    run_id: int,
    evaluation_id: int | None,
    reviewer_name: str,
    final_doctor_classification: str | None,
    final_doctor_assignment_type: str | None,
    final_nurse_classification: str | None,
    final_nurse_assignment_type: str | None,
    evidence_insufficient: bool,
    reviewer_note: str,
    final_sentence_type: str | None = None,
    final_sentence_custodial_months: float | None = None,
) -> int:
    conn.execute("UPDATE human_reviews SET is_latest = 0 WHERE run_id = ?", (run_id,))
    cur = conn.execute(
        """
        INSERT INTO human_reviews
            (run_id, evaluation_id, reviewer_name, final_doctor_classification, final_doctor_assignment_type,
             final_nurse_classification, final_nurse_assignment_type, final_sentence_type,
             final_sentence_custodial_months, evidence_insufficient, reviewer_note, is_latest, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        """,
        (run_id, evaluation_id, reviewer_name, final_doctor_classification, final_doctor_assignment_type,
         final_nurse_classification, final_nurse_assignment_type, final_sentence_type,
         final_sentence_custodial_months, int(evidence_insufficient), reviewer_note, now_iso()),
    )
    return cur.lastrowid


def get_latest_human_review(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM human_reviews WHERE run_id = ? AND is_latest = 1 ORDER BY id DESC LIMIT 1", (run_id,)
    ).fetchone()
    return row_to_dict(row)


def list_human_review_history(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM human_reviews WHERE run_id = ? ORDER BY created_at", (run_id,)
    ).fetchall()
    return rows_to_dicts(rows)


def list_human_reviews_for_experiment(conn: sqlite3.Connection, experiment_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT hr.* FROM human_reviews hr
        JOIN runs r ON r.id = hr.run_id
        WHERE r.experiment_id = ? AND hr.is_latest = 1
        """,
        (experiment_id,),
    ).fetchall()
    return rows_to_dicts(rows)


def get_latest_human_reviews_by_run(conn: sqlite3.Connection, experiment_id: int) -> dict[int, dict[str, Any]]:
    """run_id -> its current (is_latest) human review, from one query
    instead of one call to get_latest_human_review per run."""
    return {hr["run_id"]: hr for hr in list_human_reviews_for_experiment(conn, experiment_id)}


# ---------------------------------------------------------------------------
# Model catalog cache (discovery via OpenRouter /models)
# ---------------------------------------------------------------------------

def cache_models(conn: sqlite3.Connection, models: list[dict[str, Any]]) -> None:
    ts = now_iso()
    for m in models:
        conn.execute(
            """
            INSERT INTO model_catalog_cache (model_id, name, context_length, pricing_prompt, pricing_completion,
                                              supported_parameters_json, raw_json, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                name = excluded.name,
                context_length = excluded.context_length,
                pricing_prompt = excluded.pricing_prompt,
                pricing_completion = excluded.pricing_completion,
                supported_parameters_json = excluded.supported_parameters_json,
                raw_json = excluded.raw_json,
                fetched_at = excluded.fetched_at
            """,
            (
                m["id"], m.get("name"), m.get("context_length"),
                str((m.get("pricing") or {}).get("prompt")), str((m.get("pricing") or {}).get("completion")),
                json.dumps(m.get("supported_parameters") or []), json.dumps(m), ts,
            ),
        )


def get_cached_models(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM model_catalog_cache ORDER BY model_id").fetchall()
    return rows_to_dicts(rows)
