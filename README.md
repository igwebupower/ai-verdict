# The AI Verdict - LLM Experiment Lab

A local, working application for running repeatable, auditable experiments against multiple LLMs
(via [OpenRouter](https://openrouter.ai)), collecting their raw responses, classifying them with an
independent structured evaluator, human-reviewing that classification, and exporting an evidence-based
report + YouTube production brief.

Originally built to generate evidence for one video essay, starting with *"Do LLMs Assume Nurses Are
Women?"* - but the app itself is general-purpose, supporting more than one kind of study
(`evaluation_domain` on each experiment selects which evaluator/schema/analysis/dashboard treatment
applies). A second built-in experiment, *"The AI judge gave him a longer sentence,"* tests whether a
sentencing recommendation changes when only the defendant's name or only his statement's dialect changes -
see "Running the sentencing-bias sample experiment" below.

No conclusions are hardcoded anywhere. Every number shown on the dashboard, in the exported report, or in
the video brief is computed from actually recorded model responses and their (human-reviewable)
classifications.

## What's in here

```
app.py                         Home page: list / duplicate / open experiments
pages/
  1_Create_Experiment.py       Define an experiment: prompts, models, repetitions, params
  2_Run_Experiment.py          Plan runs, add/remove models, execute with live progress, pause/resume
  3_Results_Dashboard.py       Filterable results, proportions with 95% CI, disagreements
  4_Human_Review.py            Review raw response + automated classification side by side
  5_Export_and_Video_Brief.py  CSV / Markdown report / JSON archive / YouTube brief downloads
  6_Video_Script.py            Full read-on-camera narration script, generated from real results
  7_Blog_Post.py                Medium-ready blog post draft, generated from real results
  8_Settings.py                API key status, model catalog cache, integrity notes
core/
  db.py                        SQLite schema + all persistence (both evaluation domains)
  openrouter_client.py         Chat completions + model discovery, retry/backoff
  evaluator.py                 Independent gender-classification evaluator (prompt + orchestration)
  schema.py                    Pydantic JSON schema + validation for the gender evaluator's output
  sentencing_evaluator.py      Independent sentencing-classification evaluator (prompt + orchestration)
  sentencing_schema.py         Pydantic JSON schema + validation for the sentencing evaluator's output
  runner.py                    Turns experiment config into runs; executes; pause/resume/crash-safe;
                                dispatches each run's evaluation to the right domain's evaluator
  analysis.py                  Domain-agnostic statistics engine (categorical_rate, continuous_stats,
                                value_distribution + their grouped variants) plus both domains' thin
                                per-domain wrappers (role_summary, custodial_rate, etc.) - see
                                "A shared statistics/narrative engine" below
  export.py                    Gender domain's CSV / report / video brief / narration script / blog post /
                                JSON archive, plus the domain-agnostic variant-ranking engine
                                (_rank_variants_by_headline_rate) both domains' narrative generators share
  sentencing_export.py         Sentencing domain's own Blog Post generator, built on the shared engine
                                above with its own sentencing-specific prose
  charts.py                    Matplotlib chart builders for the dashboard
  seed.py                      The built-in "Do LLMs Assume Nurses Are Women?" experiment
  sentencing_seed.py           The built-in "The AI judge gave him a longer sentence." experiment
  ui_common.py                 Shared Streamlit helpers (env bootstrap, experiment selector, etc.)
tests/                         169 tests - see "Tests" below
```

### A shared statistics/narrative engine

Both evaluation domains share the same underlying statistical and narrative-ranking primitives rather
than each reimplementing its own counting/CI/ranking logic:

- `analysis.categorical_rate(df, field_col, positive_values)` - a Wilson-CI proportion for any
  categorical field and "which values count as positive" set. `role_summary`'s explicit/implied/any-
  assignment figures and `custodial_rate` are both one-line wrappers around this.
- `analysis.continuous_stats(df, numeric_col, mask_col=...)` - mean/median/CI for any numeric field,
  optionally restricted to rows matching a boolean mask. `custodial_sentence_length_stats` is a wrapper.
- `analysis.value_distribution(df, field_col)` - a full count-per-value breakdown for any categorical
  field. `sentence_type_distribution` is a wrapper.
- `analysis.grouped_categorical_rate` / `grouped_continuous_stats` - the same primitives broken out by
  arbitrary grouping columns (`["model_id"]`, `["model_id", "prompt_variant"]`, etc.).
- `export._rank_variants_by_headline_rate(df, prompt_versions, field_specs)` - decides whether an
  experiment has a "before vs. after" contrast worth building a two-part narrative around, and which
  variant to lead with. Both `export._rank_variants_by_assignment_rate` (gender) and
  `sentencing_export._rank_variants_by_custodial_rate` are one-line calls into this with their own
  `(field_col, positive_values)` pairs - no per-domain ranking/contrast-detection code.

A third evaluation domain's *statistics and ranking* would need zero new code - just its own field names
and positive-value sets passed into the primitives above. What's deliberately **not** shared is the
narrative prose itself (`sentencing_export.generate_blog_post` is hand-written sentencing-specific copy,
not a generic template) - forcing two very different findings through one sentence template is exactly
how a report ends up reading like nobody in particular wrote it. See `core/sentencing_export.py`'s module
docstring for the reasoning.

## Setup

Requires Python 3.11+ (tested on 3.13).

```bash
cd "AI Verdict"
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=sk-or-v1-...
```

Get an OpenRouter API key at https://openrouter.ai/keys. OpenRouter gives access to models from OpenAI,
Anthropic, Google, Meta, Mistral, and many others through one API and one key.

## Running

```bash
streamlit run app.py
```

Opens at http://localhost:8501. The SQLite database is created automatically at `data/ai_verdict.db` on
first run.

## Running the sample experiment

1. On the home page, click **Load built-in sample experiment**. This seeds "Do LLMs Assume Nurses Are
   Women?" with four immutable prompt variants: `baseline` (the exact prompt from the brief - measures
   spontaneous assumption, since nothing forces a pronoun at all), `forced_pronoun` (same scenario, but
   the model must use "he" or "she" after each person's first mention and singular "they" is disallowed -
   measures the forced-choice default instead), `role_reversed` (doctor/nurse swap which action they
   perform), and `alternative_profession` (pilot/flight attendant, to check whether any pattern is
   specific to doctor/nurse).
2. Go to **Run Experiment**, select the experiment, and add one or more models (either via **Discover
   models from OpenRouter**, which lists everything currently available through your key, or by typing a
   model id manually, e.g. `openai/gpt-4o-mini`).
3. Click **Plan runs** - this materialises every (prompt variant x model x repetition) combination as a
   row in the database and locks the prompt versions.
4. Click **Start / Resume**. Progress updates live; **Pause** stops after the current in-flight call
   finishes (nothing is left half-written). Resuming later picks up exactly where it left off - already
   completed runs are never re-sent.
5. Each successful response is automatically evaluated by an independent model (default
   `openai/gpt-4.1-mini`, configurable via `AI_VERDICT_EVALUATOR_MODEL`) using a structured
   rubric that never infers gender from profession, name, or stereotype - only from actual gendered
   language in the response, with a verbatim supporting quote required.
6. Go to **Human Review** to confirm or correct each automated classification. The original automated
   result is always preserved alongside your revision.
7. Go to **Results Dashboard** for proportions (with 95% Wilson confidence intervals and explicit sample
   sizes), filterable by model / prompt variant / review status.
8. Go to **Export and Video Brief** to download the CSV, the Markdown report, the YouTube production
   brief, and the full JSON archive.
9. Go to **Video Script** for a full, read-on-camera narration script built from the same real data -
   structured as a two-act "spontaneous vs. forced" story with a per-model breakdown when the data
   supports that contrast, or a simpler single-act structure when it doesn't.
10. Go to **Blog Post** for a Medium-ready draft - headline options, a dek, and full body copy, built
    from the same real contrast when there is one. Written to avoid the usual tells of AI-generated
    prose (hedge-everything, "moreover"/"delve into", rule-of-three lists, uniform paragraph rhythm) by
    leaning on specific real numbers and quotes instead of generic summary language.

## Running the sentencing-bias sample experiment

1. On the home page, click **Load sentencing-bias sample experiment**. This seeds *"The AI judge gave him
   a longer sentence"* - a fixed, invented burglary case (defendant pleads guilty, no previous
   convictions, £1,200 of stolen tools and a bicycle) with five immutable prompt variants, each changing
   exactly one thing relative to the `baseline` variant so a measured difference can't be blamed on a
   confound: `baseline` (defendant "James Mitchell", standard-English statement), `name_only` (identical
   case and statement, defendant renamed "Jamal Washington" - the Bertrand & Mullainathan 2004 name-swap
   design), and `dialect_aae` / `dialect_nigerian` / `dialect_glaswegian` (same defendant name as
   baseline - only the statement's dialect changes).
2. Steps 2-7 are the same workflow as the gender sample above (add models, plan runs, execute,
   human-review, dashboard) - the Results Dashboard and Human Review pages automatically detect the
   experiment's `evaluation_domain` and show custodial rate / sentence length instead of gender
   assignment rates.
3. The headline comparison is **By prompt variant** on the Results Dashboard: compare `name_only` against
   `baseline` for a pure name effect, and each `dialect_*` variant against `baseline` for a pure dialect
   effect - never combine two variants' effects into one claim, since each only isolates one variable.
4. **Blog Post** has its own sentencing-domain generator (`core/sentencing_export.py`) - it adapts its
   structure depending on whether one model dominates every custodial recommendation (the common case) or
   several do, and names any model with only a handful of stray results explicitly rather than folding it
   into either group. **Export and Video Brief** and **Video Script** are still gender-domain only for now
   (see "Known limitations") - for those, read results from the Results Dashboard's "Raw results table"
   expander, or query `data/ai_verdict.db` directly.

## Environment variables

See `.env.example`. All are read via `python-dotenv` from a local `.env` file - never hardcoded, never
logged, never shown in the UI (the Settings page deliberately never displays even a masked key, since
this app is designed to be screen-recorded for video production).

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes, to run experiments | Your OpenRouter API key |
| `OPENROUTER_SITE_URL` | No | Sent as `HTTP-Referer` for OpenRouter's attribution |
| `OPENROUTER_APP_NAME` | No | Sent as `X-Title` for OpenRouter's attribution |
| `AI_VERDICT_DB_PATH` | No | Overrides the SQLite file location (default `./data/ai_verdict.db`) |
| `AI_VERDICT_EVALUATOR_MODEL` | No | Default evaluator model (default `openai/gpt-4.1-mini`) |

## Analysis methodology (what the numbers mean)

Every experiment has an `evaluation_domain` (`gender_healthcare` or `sentencing_bias`) that selects which
evaluator, schema, analysis functions, and dashboard view apply - set once at creation and immutable
afterward, same as a prompt version. The two domains share the generic run/evaluation/human-review/pause-
resume machinery below but measure different things: the gender domain classifies pronoun/title usage for
"doctor" and "nurse"; the sentencing domain classifies a recommended sentence_type and, when custodial or
suspended, its length in months. The principles below apply to both unless noted.

- **Denominator = valid classified responses only.** A run only counts toward a percentage if the API
  call succeeded *and* there is either a valid automated evaluation or a human review that resolved it.
  Failed API calls and unresolved-invalid evaluations are excluded and reported separately as counts, not
  silently dropped.
- **Explicit and implied assignments are always reported separately**, never combined into one
  unexplained score. A clearly-labelled "any assignment (explicit + implied)" figure is also shown,
  explicitly defined as their sum.
- **No cross-model "bias score" is computed anywhere.** The spec for this project explicitly prohibits
  inventing one without a defensible method, so none exists in this codebase.
- **Confidence intervals use the Wilson score interval** (valid at small n, unlike the naive normal
  approximation), implemented from scratch with no extra statistics dependency.
- **Groups with fewer than 20 valid responses are labelled "exploratory"** everywhere they appear
  (dashboard, report, brief) - they are never presented as if they were statistically conclusive.
- **Repeated samples from the same model are not independent observations.** This is stated explicitly in
  every generated report. A single prompt, however many times it's run, cannot establish a general
  stereotype - that's why the sample experiment ships with role-reversed and alternative-profession
  variants, and the generated report/brief actively encourage checking all of them before concluding
  anything.

## The evaluator

The full evaluator system prompt lives in `core/evaluator.py` (`EVALUATOR_SYSTEM_PROMPT`) and is
completely separate from whatever prompt is sent to the model under test - the evaluator instructions are
never seen by, or mixed into, the tested model's call. By default a different model evaluates than
whatever is under test (configurable per-experiment).

Every evaluation must satisfy a Pydantic schema (`core/schema.py`) before being accepted:
- `gender_classification`: `explicit_male` / `explicit_female` / `explicit_other` / `unspecified` /
  `ambiguous`
- `assignment_type`: `explicit` / `implied` / `none`
- a **verbatim** supporting quote is required whenever `assignment_type != "none"` - the quote is checked
  as an actual substring of the response being evaluated, so a fabricated or paraphrased quote fails
  validation
- `ambiguous` can never pair with `explicit`/`implied` (an ambiguous reference is never counted as a
  confirmed assignment) - enforced as a hard schema rule, not a convention
- a top-level `uncertain` flag (with required `uncertain_reason` when true) lets the evaluator flag its
  own low confidence independent of any single role's classification

An invalid evaluator response gets one automatic corrective retry (the validation error is sent back to
the evaluator model); if still invalid, it is stored as `invalid_needs_review` and surfaced on the
dashboard and in Human Review rather than silently accepted or discarded.

## Experimental integrity safeguards

- The same prompt text and generation parameters are sent to every selected model; if OpenRouter's model
  catalog says a model doesn't support a requested parameter, that's recorded per-run (not silently
  dropped) in each run's `unsupported_params`.
- Prompt versions lock the moment the first run against them is planned. Editing a prompt after that
  creates a new version (`core/db.py: edit_prompt_version`); the old version is never mutated.
- Every run stores: experiment id, prompt version, model id, run number, full prompt, full raw response,
  timestamp, generation parameters, token usage, cost (when OpenRouter reports it), latency, and any
  error - successes and failures are different `status` values, never conflated.
- Retries use exponential backoff and only re-send the HTTP request when no response was already
  received for that attempt, to avoid double-charging on a slow-but-successful call.
- Runs can be paused and safely resumed; a process crash mid-call leaves a run in `running` with no
  stored response, which is detected and safely re-queued (never skipped, never duplicated) - see
  `tests/test_runner_resume.py`.
- Run execution order can be randomised (`core/runner.py: _order_pending_runs`) to avoid confounding
  model/prompt effects with time-of-run effects; the originally *planned* order is kept separately from
  the *executed* order for auditability.
- OpenRouter does not always guarantee a fixed, reproducible model snapshot behind a model id - this is
  stated explicitly in every generated report, and the exact `model` string OpenRouter actually returned
  for each call is recorded and exported.

## Tests

```bash
pytest
# or, for a coverage summary:
pytest -v
```

169 tests, no live API calls (everything that would call OpenRouter is mocked via `monkeypatch`). Covers:

- prompt immutability / versioning (`test_db.py`)
- response persistence, success vs. failure distinction (`test_db.py`)
- API retry/backoff behaviour, which errors are/aren't retried, retry exhaustion (`test_openrouter_client.py`)
- evaluator JSON schema validation, including every gender-classification edge case - ambiguous rows,
  fabricated quotes, missing uncertainty reasons (`test_schema_evaluation.py`)
- the evaluator's repair-retry loop and markdown-fence stripping (`test_evaluator.py`)
- percentage/denominator correctness, Wilson CI behaviour, exploratory-sample flagging, disagreement
  detection (`test_analysis.py`)
- human-review overrides and revision history, preserving the original automated result (`test_db.py`)
- CSV / Markdown report / video brief / JSON archive generation, including the "never fabricate an
  example" fallback (`test_export.py`)
- pause/resume, crash recovery, idempotent run planning, no duplicate API calls across a pause/resume
  boundary (`test_runner_resume.py`)
- sentencing-domain schema validation, including the custodial-months/sentence-type consistency rules
  (`test_sentencing_schema.py`)
- the sentencing evaluator's repair-retry loop and truncation handling (`test_sentencing_evaluator.py`)
- custodial rate, sentence-length stats (including n=0/n=1 edge cases), per-variant breakdowns, and human-
  review overrides in the sentencing domain (`test_sentencing_analysis.py`)
- runner domain dispatch - a sentencing-bias experiment's runs are evaluated by the sentencing evaluator
  and never touch the gender evaluator or its columns (`test_sentencing_runner_dispatch.py`)
- the built-in sentencing experiment's one-variable-at-a-time design: `name_only` differs from `baseline`
  in exactly the defendant's name, each `dialect_*` variant differs in exactly the statement, case facts
  are byte-identical across all five variants, and the stolen value is in pounds (`test_sentencing_seed.py`)
- the domain-agnostic statistics primitives directly - categorical_rate, continuous_stats,
  value_distribution and their grouped variants, plus a regression guard confirming role_summary's
  numbers didn't change when it was rewritten to use them (`test_generic_analysis.py`)
- the sentencing Blog Post generator: never fabricates a percentage (every number is checked against
  `analysis.summarize_sentencing_by`'s own output), multi-line quoted responses get `>` on every line
  (not just the first), and a model with only a stray custodial result or two is named explicitly rather
  than silently folded into either the dominant model's story or the fully-silent group
  (`test_sentencing_export.py`)
- the built-in name/nationality experiment's one-variable-at-a-time design: every variant differs from
  baseline in exactly the defendant's name, the statement's wording is byte-identical across all four
  (no dialect confound this time), and neither of the two real identifiable names from an earlier draft
  ever appears in a seeded prompt (`test_sentencing_seed.py`)
- every matplotlib chart function never leaks a Figure into pyplot's global registry across repeated
  calls, and a closed figure still renders correctly - a regression guard for a real production incident
  (see "Known limitations") (`test_charts.py`)

Fixtures for testing the evaluator without live calls live in `tests/fixtures/sample_responses.py` -
hand-written exemplar model outputs and matching evaluator JSON, used to exercise the full pipeline
deterministically.

## Known limitations

- **Hardcoded model IDs can go stale.** The original default evaluator model
  (`anthropic/claude-3.5-haiku`) was deprecated by OpenRouter between when this app was built and when it
  was first run live, causing every evaluation in the first full run to fail with "no provider serves this
  model anymore." The default is now `openai/gpt-4.1-mini`, but the underlying risk doesn't go away: any
  model id typed into an experiment (tested or evaluator) can be deprecated later. Before trusting a run's
  results, check the Results Dashboard's failed/invalid counts and the Human Review page, not just the
  top-line percentages - a 0% result across the board is a signal to check for silent evaluation failures,
  not necessarily a finding.
- Model discovery, cost reporting, and the exact set of `supported_parameters` all depend on what
  OpenRouter's `/models` endpoint currently returns for a given model - this can change over time and
  varies by provider.
- The evaluator is itself an LLM and can be wrong; that is exactly why Human Review exists and why
  automated output is never treated as ground truth anywhere in the dashboard, exports, or reports.
- This is a local, single-user tool (SQLite, short-lived connections per Streamlit rerun). It is not
  designed for concurrent multi-user access.
- **CSV/Markdown export, the video brief, and the narration script are all gender-domain only.** Each
  checks the experiment's `evaluation_domain` and refuses to run against a sentencing-bias experiment
  (rather than silently generating a nonsensical "0% gender assignment" report) - that reporting hasn't
  been built for the sentencing domain yet. The **Blog Post** generator is the exception - it has its own
  sentencing-domain version (`core/sentencing_export.py`). For the other three, use the Results
  Dashboard's "Raw results table" or query `data/ai_verdict.db` directly in the meantime.
- **(Fixed) Matplotlib figures were leaking into pyplot's global state.** `core/charts.py`'s three chart
  functions called `plt.subplots()` but never `plt.close(fig)`, so every chart ever rendered stayed
  registered in matplotlib's global figure manager for the life of the process. In a long-running
  Streamlit server that regenerates several charts on every single rerun (every filter change, every
  navigation), this surfaced live as `MemoryError: bad allocation` on the Results Dashboard after enough
  reruns accumulated (~650-700MB+ private memory observed on a process that should sit well under
  200MB). Fixed by calling `plt.close(fig)` right before each function returns its figure - this only
  unregisters it from pyplot's global state, the returned `Figure` object itself still renders fine via
  `st.pyplot(fig)`. Also forced the `Agg` backend explicitly in `core/charts.py` rather than relying on
  matplotlib's auto-detection, since that auto-detection picked an interactive GUI backend during test
  runs on this machine and failed outright (`core` modules should never depend on a GUI toolkit being
  present). Regression-tested in `test_charts.py`.
- **(Fixed) "Cost so far" never counted the evaluator's own API calls.** Every run makes two LLM calls -
  the tested model, then the independent evaluator model that judges the response - but the `evaluations`
  table had no `cost_usd` column, so every cost metric across the Run Experiment page, Results Dashboard,
  and exports only ever summed the tested model's spend. This was invisible in the UI (the numbers always
  looked plausible, just quietly incomplete) and only surfaced by cross-checking the app's displayed total
  against OpenRouter's own `/api/v1/credits` balance, which didn't reconcile. Fixed by adding `cost_usd` to
  `evaluations`, accumulating it across repair retries in `evaluator.py` and `sentencing_evaluator.py`
  (truncation and JSON-repair attempts each cost money even when the final parse succeeds), and
  propagating it through `runner.py` into `analysis.cost_summary()`, which now reports
  `tested_model_cost_usd` and `evaluation_cost_usd` separately as well as combined. Evaluations run before
  this fix have no recorded generation id, so their cost genuinely cannot be recovered - they're reported
  as `evaluations_missing_cost` (an honest "unknown"), never backfilled with an assumed $0 or an estimate,
  consistent with this project's rule of never fabricating data. Regression-tested in `test_evaluator.py`,
  `test_sentencing_evaluator.py`, and `test_analysis.py`.
