from __future__ import annotations

from core import db, export, sentencing_export


def _make_experiment(db_path, repetitions=1):
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(
            conn, name='"The test case."', hypothesis="h", repetitions=repetitions,
            evaluation_domain="sentencing_bias",
        )
        baseline_id = db.create_prompt_version(conn, exp_id, "baseline", "Recommend a sentence for James.")
        nigerian_id = db.create_prompt_version(conn, exp_id, "dialect_nigerian", "Recommend a sentence for James, Nigerian dialect.")
        db.add_experiment_model(conn, exp_id, "model-a")
        db.add_experiment_model(conn, exp_id, "model-b")
    return exp_id, baseline_id, nigerian_id


def _add_run(db_path, exp_id, pv_id, model_id, run_number, response_text, sentence_type, months=None, valid=True):
    with db.get_connection(db_path) as conn:
        run_id = db.create_run(conn, exp_id, pv_id, model_id, run_number, run_number, {})
        db.mark_run_success(conn, run_id, response_text, "gen-1", model_id, "stop", 10, 5, 15, 0.0001, 100, 0)
        eval_id = db.create_evaluation(conn, run_id, "openai/gpt-4.1-mini", "v1")
        if valid:
            db.update_sentencing_evaluation_result(
                conn, eval_id, raw_evaluator_response="{}", is_valid=True, validation_errors=None,
                parsed={"sentence_type": sentence_type, "custodial_months": months,
                        "quote": "x", "reasoning": "r"},
            )
        else:
            db.mark_evaluation_error(conn, eval_id, "bad")
    return run_id


def test_empty_experiment_returns_a_placeholder_not_a_crash(db_path):
    exp_id, _, _ = _make_experiment(db_path)
    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id)
    assert "No classified results are available yet" in post


def test_single_model_story_is_built_when_one_model_dominates_custodial_recommendations(db_path):
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=5)
    # model-a: 100% custodial at baseline, drops at the dialect variant.
    for i in range(1, 6):
        _add_run(db_path, exp_id, baseline_id, "model-a", i, "A suspended custodial sentence of 6 months.",
                  "suspended_custodial", 6)
    for i in range(1, 6):
        sentence_type = "community_order" if i <= 3 else "suspended_custodial"
        months = None if sentence_type == "community_order" else 6
        _add_run(db_path, exp_id, nigerian_id, "model-a", i, "A community order." if sentence_type == "community_order"
                  else "A suspended custodial sentence of 6 months.", sentence_type, months)
    # model-b: never recommends custody at all.
    for i in range(1, 6):
        _add_run(db_path, exp_id, baseline_id, "model-b", i, "A fine.", "fine")
        _add_run(db_path, exp_id, nigerian_id, "model-b", i, "A fine.", "fine")

    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id)

    assert "`model-a`" in post
    assert "model-b" in post  # named as one of the silent models
    assert "It's actually about one of them" in post
    assert export.word_count(post) > 50


def test_never_fabricates_custodial_rate_numbers(db_path):
    """Every percentage printed for the dominant model must match what
    analysis.summarize_sentencing_by actually computes - not a number the
    generator invented or hardcoded."""
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=4)
    for i in range(1, 5):
        _add_run(db_path, exp_id, baseline_id, "model-a", i, "A suspended custodial sentence of 6 months.",
                  "suspended_custodial", 6)
    for i in range(1, 5):
        sentence_type = "community_order" if i <= 3 else "suspended_custodial"
        months = None if sentence_type == "community_order" else 6
        _add_run(db_path, exp_id, nigerian_id, "model-a", i, "x", sentence_type, months)

    with db.get_connection(db_path) as conn:
        from core import analysis
        rows = analysis.build_sentencing_rows(conn, exp_id)
        import pandas as pd
        df = pd.DataFrame(rows)
        countable = df[df["is_countable"] & (df["model_id"] == "model-a")]
        by_variant = analysis.summarize_sentencing_by(countable, ["prompt_variant"]).set_index("prompt_variant")
        post = sentencing_export.generate_blog_post(conn, exp_id)

    baseline_pct = by_variant.loc["baseline", "custodial_pct"]
    nigerian_pct = by_variant.loc["dialect_nigerian", "custodial_pct"]
    assert f"{baseline_pct:.0f}%" in post
    assert f"{nigerian_pct:.0f}%" in post


def test_blockquoted_multiline_response_prefixes_every_line(db_path):
    """A raw response with embedded newlines (markdown headers, bullet
    points) must get '> ' on every line, not just the first - otherwise it
    falls out of the markdown blockquote after the first line break."""
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=1)
    multiline_response = "# SENTENCING RECOMMENDATION\n\n**Category: Suspended Custodial Sentence**\n\nSome reasoning here."
    _add_run(db_path, exp_id, baseline_id, "model-a", 1, multiline_response, "suspended_custodial", 6)
    _add_run(db_path, exp_id, nigerian_id, "model-a", 1, "# SENTENCING RECOMMENDATION\n\n## Category: Community Order\n\nOther reasoning.",
              "community_order")

    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id)

    for line in multiline_response.split("\n"):
        if line.strip():
            assert f"> {line}" in post, f"line not properly blockquoted: {line!r}"


def test_minor_model_is_named_not_silently_dropped(db_path):
    """A model with just one or two stray custodial recommendations must be
    named explicitly, never silently folded into either the dominant
    model's story or the fully-silent group - that would misrepresent what
    actually happened."""
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=5)
    for i in range(1, 6):
        _add_run(db_path, exp_id, baseline_id, "model-a", i, "A suspended custodial sentence of 6 months.",
                  "suspended_custodial", 6)
    for i in range(1, 6):
        sentence_type = "community_order" if i <= 3 else "suspended_custodial"
        months = None if sentence_type == "community_order" else 6
        _add_run(db_path, exp_id, nigerian_id, "model-a", i, "x", sentence_type, months)
    # model-b recommends custody exactly once - a minor/stray result.
    _add_run(db_path, exp_id, baseline_id, "model-b", 1, "A suspended custodial sentence of 3 months.",
              "suspended_custodial", 3)
    for i in range(2, 6):
        _add_run(db_path, exp_id, baseline_id, "model-b", i, "A fine.", "fine")
    for i in range(1, 6):
        _add_run(db_path, exp_id, nigerian_id, "model-b", i, "A fine.", "fine")

    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id)

    assert "`model-b` (1 time)" in post
    assert "too few to say anything about a pattern" in post


def test_evidence_insufficient_rows_are_mentioned_with_accurate_resolution(db_path):
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=5)
    for i in range(1, 6):
        _add_run(db_path, exp_id, baseline_id, "model-a", i, "A suspended custodial sentence of 6 months.",
                  "suspended_custodial", 6)
    for i in range(1, 6):
        sentence_type = "community_order" if i <= 3 else "suspended_custodial"
        months = None if sentence_type == "community_order" else 6
        _add_run(db_path, exp_id, nigerian_id, "model-a", i, "x", sentence_type, months)

    run_id = _add_run(db_path, exp_id, baseline_id, "model-b", 1, "truncated", "fine")
    with db.get_connection(db_path) as conn:
        evaluation = db.get_latest_evaluation_for_run(conn, run_id)
        db.create_human_review(
            conn, run_id, evaluation["id"], "reviewer", None, None, None, None, True,
            "truncated response", final_sentence_type=None, final_sentence_custodial_months=None,
        )
    for i in range(2, 6):
        _add_run(db_path, exp_id, baseline_id, "model-b", i, "A fine.", "fine")
    for i in range(1, 6):
        _add_run(db_path, exp_id, nigerian_id, "model-b", i, "A fine.", "fine")

    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id)

    assert "thrown out entirely" in post
    assert "model-b" in post


def _add_many(db_path, exp_id, pv_id, model_id, n, custodial_count, start_run_number=1):
    """Creates n countable runs for one variant, the first custodial_count of
    them custodial, the rest community_order - for building Wilson-CI-sized
    samples without 60 lines of individual _add_run calls."""
    for i in range(n):
        sentence_type = "suspended_custodial" if i < custodial_count else "community_order"
        months = 6 if sentence_type == "suspended_custodial" else None
        _add_run(db_path, exp_id, pv_id, model_id, start_run_number + i, "x", sentence_type, months)


def test_confound_check_paragraph_when_control_rules_out_formality(db_path):
    """When dialect_formal_control's custodial rate matches baseline (not
    the Nigerian-English drop), the post must say formality doesn't explain
    the finding - not claim the confound was never checked."""
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    with db.get_connection(db_path) as conn:
        control_id = db.create_prompt_version(conn, exp_id, "dialect_formal_control", "Recommend a sentence, formal control.")

    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)       # 90%
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 2)        # 10% - the real drop
    _add_many(db_path, exp_id, control_id, "model-a", 20, 18)        # 90% - matches baseline

    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id)

    assert "formal control" in post
    assert "doesn't explain the drop" in post
    assert "rule out the simplest alternative" in post


def test_confound_check_paragraph_when_control_confirms_formality_effect(db_path):
    """When dialect_formal_control ALSO drops like Nigerian English does,
    the post must say the honest framing has to shift - never silently keep
    the original "it's about Nigerian English" claim once it's undermined."""
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    with db.get_connection(db_path) as conn:
        control_id = db.create_prompt_version(conn, exp_id, "dialect_formal_control", "Recommend a sentence, formal control.")

    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)       # 90%
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 2)        # 10%
    _add_many(db_path, exp_id, control_id, "model-a", 20, 3)         # 15% - also dropped

    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id)

    assert "formal control also dropped" in post
    assert "honest framing" in post


def test_companion_paragraph_included_for_a_genuine_null_companion(db_path):
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 2)

    with db.get_connection(db_path) as conn:
        companion_id = db.create_experiment(
            conn, name="Companion", repetitions=10, evaluation_domain="sentencing_bias",
        )
        comp_pv1 = db.create_prompt_version(conn, companion_id, "baseline", "x")
        comp_pv2 = db.create_prompt_version(conn, companion_id, "name_other", "y")
        db.add_experiment_model(conn, companion_id, "model-a")
    _add_many(db_path, companion_id, comp_pv1, "model-a", 10, 5)   # 50%
    _add_many(db_path, companion_id, comp_pv2, "model-a", 10, 5)   # 50% - identical, genuine null

    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id, companion_experiment_id=companion_id)

    assert "Worth mentioning" in post
    assert "Nothing moved there either" in post


def test_companion_paragraph_omitted_when_companion_shows_a_real_effect(db_path):
    """Must never claim a null result for the companion if it actually found
    something - that would misrepresent the companion experiment's own data."""
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 2)

    with db.get_connection(db_path) as conn:
        companion_id = db.create_experiment(
            conn, name="Companion", repetitions=20, evaluation_domain="sentencing_bias",
        )
        comp_pv1 = db.create_prompt_version(conn, companion_id, "baseline", "x")
        comp_pv2 = db.create_prompt_version(conn, companion_id, "name_other", "y")
        db.add_experiment_model(conn, companion_id, "model-a")
    _add_many(db_path, companion_id, comp_pv1, "model-a", 20, 18)   # 90%
    _add_many(db_path, companion_id, comp_pv2, "model-a", 20, 2)    # 10% - a real effect in the companion too

    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id, companion_experiment_id=companion_id)

    assert "Worth mentioning" not in post


def test_companion_paragraph_omitted_when_companion_has_no_data(db_path):
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 2)

    with db.get_connection(db_path) as conn:
        empty_companion_id = db.create_experiment(conn, name="Empty companion", evaluation_domain="sentencing_bias")

    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id, companion_experiment_id=empty_companion_id)

    assert "Worth mentioning" not in post


def test_companion_paragraph_omitted_when_no_companion_id_given(db_path):
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 2)

    with db.get_connection(db_path) as conn:
        post = sentencing_export.generate_blog_post(conn, exp_id)

    assert "Worth mentioning" not in post


# ---------------------------------------------------------------------------
# generate_full_script
# ---------------------------------------------------------------------------

def test_script_empty_experiment_returns_a_placeholder_not_a_crash(db_path):
    exp_id, _, _ = _make_experiment(db_path)
    with db.get_connection(db_path) as conn:
        script = sentencing_export.generate_full_script(conn, exp_id)
    assert "No classified results are available yet" in script


def test_script_single_model_story_covers_cold_open_and_acts(db_path):
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 2)

    with db.get_connection(db_path) as conn:
        script = sentencing_export.generate_full_script(conn, exp_id)

    assert "[COLD OPEN - 0:00]" in script
    assert "`model-a`" in script
    assert "[ACT 1 - WHAT `MODEL-A` ACTUALLY DID]" in script
    assert "[ACT 3" in script
    assert "[PRODUCTION CHECKLIST]" in script
    assert export.word_count(script) > 50


def test_script_cold_open_uses_the_dominant_models_own_ranking_not_the_pooled_one(db_path):
    """Regression test for a real bug: with a minor second model present,
    the POOLED custodial rate per variant can disagree with what the
    dominant model itself did (a small, opposite-direction contribution
    from the minor model is enough to flip which variant looks "lowest").
    The cold open narrates what the dominant model did, so it must rank
    variants using the dominant model's own rates, not the pooled ones -
    otherwise it can claim a "drop" into a variant that model actually
    scored HIGHER on. model-a (dominant) goes 90% -> 100% baseline ->
    dialect_nigerian (no real within-model contrast); model-b (minor)
    goes 20% -> 0%, which pulls the POOLED dialect_nigerian rate below
    the pooled baseline rate even though model-a's own didn't drop."""
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 20)
    _add_many(db_path, exp_id, baseline_id, "model-b", 20, 4)
    _add_many(db_path, exp_id, nigerian_id, "model-b", 20, 0)

    with db.get_connection(db_path) as conn:
        script = sentencing_export.generate_full_script(conn, exp_id)

    assert "except rewritten in" not in script, \
        "model-a's own rate didn't drop for any variant - there is no real contrast to narrate as a drop"
    assert "Not harsher. Lighter." not in script
    assert "90%" in script and "100%" in script


def test_script_never_fabricates_custodial_rate_numbers(db_path):
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 2)

    with db.get_connection(db_path) as conn:
        from core import analysis
        import pandas as pd
        rows = analysis.build_sentencing_rows(conn, exp_id)
        df = pd.DataFrame(rows)
        countable = df[df["is_countable"] & (df["model_id"] == "model-a")]
        by_variant = analysis.summarize_sentencing_by(countable, ["prompt_variant"]).set_index("prompt_variant")
        script = sentencing_export.generate_full_script(conn, exp_id)

    assert f"{by_variant.loc['baseline', 'custodial_pct']:.0f}%" in script
    assert f"{by_variant.loc['dialect_nigerian', 'custodial_pct']:.0f}%" in script


def test_script_confound_check_when_control_rules_out_formality(db_path):
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    with db.get_connection(db_path) as conn:
        control_id = db.create_prompt_version(conn, exp_id, "dialect_formal_control", "Recommend a sentence, formal control.")
    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 2)
    _add_many(db_path, exp_id, control_id, "model-a", 20, 18)

    with db.get_connection(db_path) as conn:
        script = sentencing_export.generate_full_script(conn, exp_id)

    assert "[ACT 2 - CHECKING MY OWN FINDING]" in script
    assert "doesn't explain this" in script
    assert "ALSO dropped" not in script


def test_script_confound_check_when_control_confirms_formality_effect(db_path):
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=20)
    with db.get_connection(db_path) as conn:
        control_id = db.create_prompt_version(conn, exp_id, "dialect_formal_control", "Recommend a sentence, formal control.")
    _add_many(db_path, exp_id, baseline_id, "model-a", 20, 18)
    _add_many(db_path, exp_id, nigerian_id, "model-a", 20, 2)
    _add_many(db_path, exp_id, control_id, "model-a", 20, 3)

    with db.get_connection(db_path) as conn:
        script = sentencing_export.generate_full_script(conn, exp_id)

    assert "ALSO dropped" in script
    assert "honest twist" in script


def test_script_falls_back_to_simple_readout_when_no_model_dominates(db_path):
    """When no single model accounts for the custodial pattern, the script
    must not force the one-model Act 1/2/3 structure onto data that doesn't
    support it."""
    exp_id, baseline_id, nigerian_id = _make_experiment(db_path, repetitions=10)
    _add_many(db_path, exp_id, baseline_id, "model-a", 10, 5)
    _add_many(db_path, exp_id, nigerian_id, "model-a", 10, 5)
    _add_many(db_path, exp_id, baseline_id, "model-b", 10, 5)
    _add_many(db_path, exp_id, nigerian_id, "model-b", 10, 5)

    with db.get_connection(db_path) as conn:
        script = sentencing_export.generate_full_script(conn, exp_id)

    assert "[ACT 1" not in script
    assert "[THE RESULTS]" in script
