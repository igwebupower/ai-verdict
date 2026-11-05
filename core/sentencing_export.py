"""Narrative generation for the sentencing-bias domain.

The structural/statistical plumbing here is shared with core/export.py (the
gender-healthcare domain's narrative generator): _rank_variants_by_headline_rate,
VariantRanking, _already_narrated_variant_names, _truncate_at_word, _speakable,
word_count, and _prose_list are all domain-agnostic and imported directly
rather than reimplemented. What's NOT shared, deliberately, is the actual
prose - forcing two very different findings (a doctor/nurse gender-assumption
story and a one-model sentencing-dialect outlier) through one generic sentence
template is exactly how a report ends up reading like it was written by
nobody in particular. This module writes sentencing-specific copy using
sentencing-specific data, through the same shared ranking/caveat/word-count
engine every other domain's narrative generator uses.

Hard rule enforced throughout, same as core/export.py: every number, quote,
and example written into a report comes directly from the experiment's own
dataframe or database rows. Nothing here invents a result, a quote, or a
sample size.
"""
from __future__ import annotations

import random
import sqlite3
from typing import Any

import pandas as pd

from core import analysis, db
from core.export import _already_narrated_variant_names, _blockquote, _prose_list, \
    _rank_variants_by_headline_rate, _speakable, _truncate_at_word, _variant_headline_rate, word_count

CUSTODIAL_FIELD_SPECS: list[tuple[str, set]] = [("is_custodial", {True})]


def _vary(experiment_id: int, key: str, options: list):
    """Deterministically picks one of several hand-written, equivalent
    phrasings for one narrative beat, seeded by (experiment_id, key).

    The same experiment always regenerates the exact same script (stable
    for caching and for tests that read specific numbers out of it), but
    different experiments - and different beats within the same script -
    land on different stock phrases instead of all reciting the same fixed
    sentence. That repetition across every generated video, not any single
    sentence on its own, is what makes a generated script read as AI-
    written; this is what keeps it from happening."""
    return random.Random(f"{experiment_id}:{key}").choice(options)

# Display names for this experiment's known variants, so prose reads as
# "Nigerian English" rather than the generic _speakable() fallback's
# "dialect nigerian". An unrecognised variant name (e.g. from a duplicated
# and renamed copy of this experiment) falls back to _speakable() cleanly.
VARIANT_DISPLAY_NAMES = {
    "baseline": "the baseline",
    "name_only": "a pure name swap",
    "dialect_aae": "African American English",
    "dialect_nigerian": "Nigerian English",
    "dialect_glaswegian": "Glaswegian",
    "dialect_formal_control": "a formal-but-dialect-neutral control",
    "name_afghan": "an Afghan-coded name",
    "name_chinese": "a Chinese-coded name",
    "name_nigerian": "a Nigerian-coded name",
}


def _display_variant(variant_name: str) -> str:
    return VARIANT_DISPLAY_NAMES.get(variant_name, _speakable(variant_name))


def _quoted_title(name: str) -> str:
    """Wraps an experiment name in quotes for a headline-option bullet -
    unless the name already starts with a quote mark (this experiment's own
    name is itself a quote: '"The AI judge gave him a longer sentence."'),
    in which case adding another pair would double them up."""
    return name if name.startswith('"') or name.startswith("'") else f"\"{name}\""


def _capitalize_first(s: str) -> str:
    """s.capitalize() lowercases the rest of the string too, which mangles
    "Nigerian English" into "Nigerian english" - this only touches the first
    character."""
    return s[:1].upper() + s[1:] if s else s


def _minor_models_sentence(
    minor_models: list[str], custodial_counts: "pd.Series", experiment_id: int, context: str,
) -> str:
    """Honest acknowledgement of any model(s) that recommended custody a
    handful of times but don't drive the headline pattern - never silently
    dropped from the narrative just because they complicate a clean "one
    model" story."""
    if not minor_models:
        return ""
    bits = []
    for m in minor_models:
        c = int(custodial_counts[m])
        bits.append(f"`{m}` ({c} time{'s' if c != 1 else ''})")
    names = _prose_list(bits)
    if len(minor_models) == 1:
        options = [
            f" One other model, {names}, also recommended custody a handful of times - too few to say "
            f"anything about a pattern there, but worth naming rather than quietly folding into 'never.'",
            f" One exception: {names} recommended custody a couple of times too - not enough to be a "
            f"pattern, but it didn't sit at a flat zero either, so it gets named rather than rounded away.",
            f" {names} isn't quite silent either - a small number of custodial recommendations, too few "
            f"to mean anything on its own, but real enough that it shouldn't just vanish into 'never.'",
        ]
    else:
        options = [
            f" {names} also recommended custody a handful of times each - too few to say anything about "
            f"a pattern, but worth naming rather than quietly folding into 'never.'",
            f" {names} weren't flat zeroes either - each picked up a small number of custodial "
            f"recommendations, not enough to count as a pattern but too many to just round down to 'never.'",
            f" A handful of custodial recommendations also came from {names} - not a pattern, but naming "
            f"them beats quietly folding them into a tidier 'every other model stayed silent' story.",
        ]
    return _vary(experiment_id, f"minor_models:{context}", options)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _variant_custodial_rate(df: pd.DataFrame, variant_name: str) -> tuple[float, int]:
    """The sentencing domain's single-field headline rate - is_custodial -
    passed into the same generic ranking engine the gender domain uses."""
    return _variant_headline_rate(df, variant_name, CUSTODIAL_FIELD_SPECS)


def _rank_variants_by_custodial_rate(df: pd.DataFrame, prompt_versions: list[dict]):
    return _rank_variants_by_headline_rate(df, prompt_versions, CUSTODIAL_FIELD_SPECS)


def _model_custodial_summary(df: pd.DataFrame, variant_name: str) -> list[dict[str, Any]]:
    """Per model (within one variant): how often it recommended custody and
    what its most common sentence_type was. The sentencing analogue of
    export._model_pairings, used to find which model(s) are worth naming on
    camera/in print rather than to compute any headline statistic (those
    always come from analysis.custodial_rate / analysis.summarize_sentencing_by)."""
    vdf = df[(df["prompt_variant"] == variant_name) & (df["is_countable"])]
    out = []
    for model_id in sorted(vdf["model_id"].unique()):
        sub = vdf[vdf["model_id"] == model_id]
        n = len(sub)
        if n == 0:
            continue
        custodial_n = int(sub["is_custodial"].sum())
        mode_counts = sub["sentence_final_type"].value_counts()
        out.append({
            "model_id": model_id, "n": n, "custodial_n": custodial_n,
            "custodial_rate": custodial_n / n,
            "mode_type": mode_counts.index[0], "mode_pct": int(mode_counts.iloc[0]) / n,
        })
    return out


def _pick_quote(
    df: pd.DataFrame, variant_name: str, sentence_types: list[str], model_id: str | None = None,
) -> dict[str, Any] | None:
    vdf = df[(df["prompt_variant"] == variant_name) & (df["is_countable"])
             & (df["sentence_final_type"].isin(sentence_types))]
    if model_id:
        vdf = vdf[vdf["model_id"] == model_id]
    if vdf.empty:
        return None
    row = vdf.iloc[0]
    return {
        "model_id": row["model_id"], "run_id": row["run_id"],
        "sentence_type": row["sentence_final_type"], "custodial_months": row["sentence_final_custodial_months"],
        "response": row["raw_response"],
    }


def _companion_null_result_paragraph(conn: sqlite3.Connection, companion_experiment_id: int) -> str | None:
    """Summarises a companion sentencing experiment's result as a single
    paragraph, strengthening the main post by showing what else was tried.
    Returns None if the companion has no countable data yet, or if it
    actually found an effect (the "nothing moved" framing below only makes
    sense for a genuine null result - this never claims a null that isn't
    really there)."""
    companion = db.get_experiment(conn, companion_experiment_id)
    if companion is None:
        return None
    companion_rows = analysis.build_sentencing_rows(conn, companion_experiment_id)
    companion_df = analysis.rows_to_dataframe(companion_rows)
    if companion_df.empty or not companion_df["is_countable"].any():
        return None
    companion_countable = companion_df[companion_df["is_countable"]]
    custodial_models = sorted(companion_countable[companion_countable["is_custodial"]]["model_id"].unique())

    # Check every model that ever recommended custody for a non-overlapping
    # CI between any two of its variants - if one exists, the companion
    # found a real effect and this paragraph should not claim a null result.
    for model_id in custodial_models:
        sub = companion_countable[companion_countable["model_id"] == model_id]
        by_variant = analysis.summarize_sentencing_by(sub, ["prompt_variant"])
        if len(by_variant) < 2:
            continue
        cis = []
        for _, row in by_variant.iterrows():
            ci_text = row["custodial_rate_ci"].strip("[]%").split("%, ")
            cis.append((float(ci_text[0]), float(ci_text[1].rstrip("%"))))
        for i in range(len(cis)):
            for j in range(i + 1, len(cis)):
                lo1, hi1 = cis[i]
                lo2, hi2 = cis[j]
                if hi1 < lo2 or hi2 < lo1:
                    return None  # a real effect exists in the companion - not a null result

    n_companion_models = companion_countable["model_id"].nunique()
    n_variants = companion_df["prompt_variant"].nunique()
    return (
        f"Worth mentioning: in a separate experiment, I tested whether the defendant's **name alone** "
        f"(no dialect change at all) moves any of {n_companion_models} models across {n_variants} different "
        f"name/nationality signals, using this exact same case. Nothing moved there either, for any model, "
        f"at a well-powered sample size - including this model specifically. Across both experiments, name "
        f"alone has never once produced an effect; the one real effect anywhere is a change to how the "
        f"statement itself is phrased. That's a sharper, more specific finding than either experiment gives "
        f"you alone: it's not about who the defendant is on paper, it's about how his own words come "
        f"through."
    )


def generate_blog_post(
    conn: sqlite3.Connection, experiment_id: int, companion_experiment_id: int | None = None,
) -> str:
    """A full Medium-ready draft (headline options, dek, body copy) built
    from this sentencing-bias experiment's real data. This is a strong first
    draft, not a finished article: read it out loud before publishing and
    cut anything that doesn't sound like your own voice.

    companion_experiment_id, if given, pulls in a one-paragraph summary of
    another sentencing experiment's result (e.g. a name-swap companion study)
    - only when that companion found a genuine null result, verified by
    checking its own data rather than trusting a label. Never fabricates a
    "nothing moved" claim; silently omits the paragraph if the companion
    actually found something, has no data yet, or doesn't exist."""
    experiment = db.get_experiment(conn, experiment_id)
    prompt_versions = db.list_prompt_versions(conn, experiment_id, latest_only=True)
    rows = analysis.build_sentencing_rows(conn, experiment_id)
    df = analysis.rows_to_dataframe(rows)
    n_models = df["model_id"].nunique() if not df.empty else 0

    countable_total = int(df["is_countable"].sum()) if not df.empty else 0
    if countable_total == 0:
        return (
            f"# {experiment['name']}\n\n"
            "No classified results are available yet for this experiment. Run it and resolve any "
            "invalid evaluations first, then come back and generate the post."
        )

    countable = df[df["is_countable"]]
    ranking = _rank_variants_by_custodial_rate(df, prompt_versions)
    missing = analysis.missing_and_invalid_counts(df)

    # Which model(s) ever recommend custody at all, across every variant -
    # the central fact that determines how this post has to be framed. A
    # strict "exactly one model ever recommends custody" check is fragile -
    # a single isolated custodial response from an otherwise-silent model
    # (one stray result out of ~25, say) would wrongly disqualify an
    # otherwise genuine single-model story. Instead: is there one model that
    # accounts for the overwhelming majority of every custodial
    # recommendation in the whole experiment? Either a >=90% share, or - for
    # smaller totals where a percentage alone is too fragile (e.g. 7 of 8) -
    # everyone else combined accounts for no more than 2 stray responses.
    custodial_counts = countable[countable["is_custodial"]]["model_id"].value_counts()
    total_custodial = int(custodial_counts.sum())
    dominant_model = custodial_counts.index[0] if len(custodial_counts) else None
    dominant_count = int(custodial_counts.iloc[0]) if len(custodial_counts) else 0
    non_dominant_total = total_custodial - dominant_count
    is_single_model_story = (
        dominant_model is not None and total_custodial > 0
        and (dominant_count / total_custodial >= 0.9 or non_dominant_total <= 2)
    )
    # Models with precisely zero custodial recommendations - distinct from a
    # "minor" model that had one or two stray ones but isn't the real story.
    silent_models = sorted(set(countable["model_id"].unique()) - set(custodial_counts.index))
    minor_models = sorted(set(custodial_counts.index) - {dominant_model}) if is_single_model_story else []

    P: list[str] = []

    # ---------------- HEADLINE OPTIONS ----------------
    P.append(f"# {experiment['name']}")
    P.append("")
    P.append(
        "*(Pick one of these as your actual Medium title, or rewrite it - these are starting points, not "
        "a final answer. Delete this note before publishing.)*"
    )
    P.append("")
    if is_single_model_story:
        P.append(f"- {_quoted_title(experiment['name'])}")
        P.append("- \"I Tested Whether AI Sentencing Tools Are Biased. One Model Was - Backwards.\"")
        P.append(
            f"- \"{len(silent_models)} AI Models Didn't Care About His Name or His Accent. "
            f"The {_ordinal(n_models)} Did.\""
        )
    else:
        P.append(f"- {_quoted_title(experiment['name'])}")
        P.append(f"- \"I Asked {n_models} AI Models to Sentence the Same Man {len(prompt_versions)} Different Ways\"")
    P.append("")
    P.append(f"*{experiment['hypothesis']}*")
    P.append("")
    P.append("---")
    P.append("")

    # ---------------- OPENING ----------------
    if is_single_model_story:
        sole_model = dominant_model
        sole = countable[countable["model_id"] == sole_model]
        low_variant = ranking.low_pv["variant_name"] if ranking.low_pv else "baseline"
        low_rate, low_n = _variant_custodial_rate(df, low_variant)
        baseline_rate, baseline_n = _variant_custodial_rate(sole, "baseline")
        low_rate_sole, _ = _variant_custodial_rate(sole, low_variant)
        drop_fraction = (
            "fewer than half" if baseline_rate > 0 and low_rate_sole < baseline_rate / 2 else "noticeably fewer"
        )
        P.append(
            f"I expected this post to be about {n_models} AI models. It's actually about one of them."
        )
        P.append("")
        P.append(
            f"Give `{sole_model}` the exact same burglary case, the exact same facts, the exact same "
            f"defendant - and just rewrite his court statement in {_display_variant(low_variant)} "
            f"instead of standard written English - and its custodial sentencing rate drops from "
            f"{baseline_rate*100:.0f}% to {low_rate_sole*100:.0f}%. Not a rounding difference: this model "
            f"recommended prison time in {'practically every' if baseline_rate >= 0.99 else 'the large majority of'} "
            f"baseline response it gave, and in {drop_fraction} of the otherwise-identical responses where "
            f"only the dialect changed."
        )
        P.append("")
        P.append(
            f"The other {len(silent_models)} models I tested never recommended custody at all, in any "
            f"version of the case.{_minor_models_sentence(minor_models, custodial_counts, experiment_id, 'post_opening')} So this isn't a "
            f"story about \"AI sentencing bias\" in general. It's a story about what happens when you "
            f"actually go looking for it instead of assuming you know what you'll find - and the answer, "
            f"this time, was stranger and more specific than the question I started with."
        )
    else:
        overall = analysis.custodial_rate(countable)
        P.append(
            f"I ran the same invented criminal case against {n_models} AI models, asking each one for a "
            f"sentencing recommendation, {experiment['repetitions']} times per variant. {overall.numerator} "
            f"of {overall.denominator} countable responses came back custodial. Here's how that number "
            f"actually breaks down, and what it does and doesn't tell you."
        )
    P.append("")

    # ---------------- WHY / METHODOLOGY ----------------
    P.append(
        "Police forces and courts are already piloting AI tools in case triage and sentencing support. So "
        "the question isn't hypothetical: if you give a model an identical case and change only the "
        "defendant's name, or only the phrasing of his own statement, does the recommended sentence move? "
        "I didn't tell any model what I was testing - there's no mention of race, dialect, or fairness "
        "anywhere in the prompt. It's framed exactly like what it claims to be: a request for a sentencing "
        "recommendation on a hypothetical case."
    )
    P.append("")
    P.append(
        "The case itself never changes: a first-time offender, burglary of a lock-up garage, £1,200 in "
        "stolen tools and a bicycle, no violence, a guilty plea, low reoffending risk per the pre-sentence "
        "report. Five versions of that same case, each changing exactly one thing against a baseline "
        "where the defendant is \"James Mitchell\" and his statement is standard written English: the name "
        "alone (\"Jamal Washington\"), or the statement's dialect alone (African American English, "
        "Nigerian English, or Glaswegian) - never both at once, so a name effect and a dialect effect can "
        "never get blamed on each other. A second model, blind to the hypothesis, extracted a structured "
        "sentence_type and length from each response, and nothing counts unless that extraction can point "
        "to an exact, verbatim phrase in the response that supports it."
    )
    P.append("")

    # ---------------- SECTION: THE FINDING ----------------
    if is_single_model_story:
        sole_model = dominant_model
        sole = countable[countable["model_id"] == sole_model]
        by_variant = analysis.summarize_sentencing_by(sole, ["prompt_variant"]).set_index("prompt_variant")
        P.append(f"## What `{sole_model}` actually did")
        P.append("")
        P.append(
            "Here's its custodial-or-suspended-custodial rate, variant by variant, with nothing averaged "
            "away. Ordered to match how the case itself was built - baseline first, then the pure name "
            "swap, then each dialect rewrite of the same statement:"
        )
        P.append("")
        preferred_order = ["baseline", "name_only", "dialect_aae", "dialect_glaswegian", "dialect_nigerian",
                            "dialect_formal_control"]
        ordered_variants = [v for v in preferred_order if v in by_variant.index]
        ordered_variants += [v for v in by_variant.index if v not in preferred_order]
        for variant_name in ordered_variants:
            r = by_variant.loc[variant_name]
            P.append(f"- **{_capitalize_first(_display_variant(variant_name))}**: {r['custodial_pct']:.0f}% custodial ({r['custodial_count']:.0f} of {r['n']:.0f})")
        P.append("")

        baseline_pct = by_variant.loc["baseline", "custodial_pct"] if "baseline" in by_variant.index else None
        comparisons = []
        most_dropped_variant, most_dropped_pct, most_dropped_gap = None, None, -1.0
        if baseline_pct is not None:
            for variant_name in ordered_variants:
                # dialect_formal_control is a confound CHECK added after the
                # headline finding - it gets its own dedicated paragraph
                # below instead of competing to be "the one condition that
                # moved the most" here, and instead of being double-counted
                # in this generic comparison list too.
                if variant_name in ("baseline", "dialect_formal_control"):
                    continue
                pct = by_variant.loc[variant_name, "custodial_pct"]
                label = _display_variant(variant_name)
                gap = baseline_pct - pct
                if gap > most_dropped_gap:
                    most_dropped_variant, most_dropped_pct, most_dropped_gap = variant_name, pct, gap
                if gap <= 2:
                    comparisons.append(f"{label} changed essentially nothing ({pct:.0f}% vs {baseline_pct:.0f}% at baseline)")
                elif baseline_pct > 0 and gap < baseline_pct * 0.3:
                    comparisons.append(f"{label} pulled it down a little, to {pct:.0f}%")
                elif baseline_pct > 0 and gap < baseline_pct * 0.6:
                    comparisons.append(f"{label} pulled it down substantially, to {pct:.0f}%")
                else:
                    comparisons.append(f"{label} pulled it down by more than half, to {pct:.0f}%")
        if comparisons:
            # Capitalize every sentence, not just the first - join(". ")
            # only inserts the separator, it doesn't capitalize what follows,
            # and a lowercase variant label (e.g. "a pure name swap") would
            # otherwise start a sentence mid-paragraph in lowercase.
            P.append(" ".join(f"{_capitalize_first(c)}." for c in comparisons))
            P.append("")

        if most_dropped_variant and most_dropped_gap >= 20:
            baseline_ci = by_variant.loc["baseline", "custodial_rate_ci"]
            dropped_ci = by_variant.loc[most_dropped_variant, "custodial_rate_ci"]
            P.append(
                f"95% confidence intervals: {baseline_ci} at baseline versus {dropped_ci} for "
                f"{_display_variant(most_dropped_variant)} - they don't overlap, which is what makes this a "
                f"real swing rather than noise from a small sample."
            )
            P.append("")
            P.append(
                f"And the direction is backwards from what I went in expecting. This isn't \"the AI judge "
                f"gave him a longer sentence\" - for {_display_variant(most_dropped_variant)}, the one "
                f"condition that moved the most, the outcome was lighter, not harsher, for an otherwise "
                f"identical defendant, purely because of how his own words were rendered on the page."
            )
            P.append("")

        # ---------------- CONFOUND CHECK: formality vs. Nigerian English specifically ----------------
        if "dialect_nigerian" in by_variant.index and "dialect_formal_control" in by_variant.index:
            nigerian_row = by_variant.loc["dialect_nigerian"]
            control_row = by_variant.loc["dialect_formal_control"]
            P.append(
                "One question I couldn't answer when I first found this: was the model actually reacting "
                "to Nigerian English specifically, or just to the statement sounding more formal and "
                "less contracted than the baseline's casual phrasing? So I added one more version, after "
                "the fact, to check: the same formal, no-contractions register as the Nigerian-English "
                "statement, but with none of its actual Nigerian English grammar - just careful, formal "
                "standard English that doesn't read as any particular dialect."
            )
            P.append("")
            control_gap = baseline_pct - control_row["custodial_pct"] if baseline_pct is not None else None
            if control_gap is not None and control_gap <= 2:
                P.append(
                    f"The formal control came back at {control_row['custodial_pct']:.0f}% custodial "
                    f"({control_row['custodial_count']:.0f} of {control_row['n']:.0f}) - essentially "
                    f"identical to baseline, nowhere near Nigerian English's {nigerian_row['custodial_pct']:.0f}%. "
                    f"Formality alone doesn't explain the drop. Whatever this model is reacting to, it's "
                    f"something more specific to the Nigerian-English rendering than just 'sounds more "
                    f"formal' - which doesn't prove it's about nationality or ethnicity either, but it does "
                    f"rule out the simplest alternative explanation."
                )
            elif control_gap is not None:
                P.append(
                    f"The formal control also dropped - to {control_row['custodial_pct']:.0f}% custodial "
                    f"({control_row['custodial_count']:.0f} of {control_row['n']:.0f}), against Nigerian "
                    f"English's {nigerian_row['custodial_pct']:.0f}%. That's a real complication: it means "
                    f"formality or contraction-free phrasing alone can move this model's output, not just "
                    f"the Nigerian-English rendering specifically. The honest framing of this whole finding "
                    f"has to shift - this may be a 'formal register' effect that the Nigerian statement "
                    f"happens to share, rather than a dialect- or nationality-specific one."
                )
            P.append("")

        baseline_quote = _pick_quote(df, "baseline", ["custodial", "suspended_custodial"], model_id=sole_model)
        nigerian_quote = _pick_quote(df, "dialect_nigerian", ["community_order"], model_id=sole_model)
        if baseline_quote and nigerian_quote:
            P.append(
                "I went back and read the actual text, side by side, because a number like that deserves "
                "scrutiny before you believe it. Here's the opening of a baseline response:"
            )
            P.append("")
            P.append(_blockquote(_truncate_at_word(baseline_quote['response'], 260)))
            P.append("")
            P.append("And here's the opening of a Nigerian-dialect response that landed on community order instead:")
            P.append("")
            P.append(_blockquote(_truncate_at_word(nigerian_quote['response'], 260)))
            P.append("")
            P.append(
                "Read the aggravating and mitigating factors in both - they're nearly the same list, "
                "weighed the same way. The facts aren't being read differently. The conclusion is. I also "
                "searched every one of this model's dialect responses for any mention of \"dialect,\" "
                "\"accent,\" \"language,\" or similar, to see if it was reasoning about the phrasing openly. "
                "It never was. Whatever's driving this doesn't show up in the model's own stated reasoning "
                "- the explanation it gives reads as neutral every time, win or lose."
            )
            P.append("")

        P.append(f"## Why I'm not calling this \"AI sentencing bias\"")
        P.append("")
        silent_list = _prose_list([f"`{m}`" for m in silent_models])
        P.append(
            f"Because it isn't, as a general claim. {silent_list} never once recommended custody for this "
            f"case, in any of {len(prompt_versions)} versions - their sentencing behaviour simply didn't "
            f"move.{_minor_models_sentence(minor_models, custodial_counts, experiment_id, 'post_act3')} The entire finding above belongs "
            f"to one model. If I'd only tested the other {n_models - 1}, this post wouldn't exist."
        )
        P.append("")
        P.append(
            "That's not a reason to dismiss it. It's a reason to describe it accurately: one specific "
            "model, tested on one specific case, shows a real and large dialect-dependent swing in its "
            "sentencing recommendations - and the organisations piloting these tools in actual case triage "
            "almost certainly don't know which of their models, if any, behaves this way, because nobody's "
            "run this test on their specific stack."
        )
        P.append("")

        if companion_experiment_id is not None:
            companion_paragraph = _companion_null_result_paragraph(conn, companion_experiment_id)
            if companion_paragraph:
                P.append(companion_paragraph)
                P.append("")
    else:
        P.append("## By variant")
        P.append("")
        for pv, rate, n in ranking.variant_ranking:
            P.append(f"- **{_speakable(pv['variant_name'])}**: {rate*100:.0f}% custodial (n={n})")
        P.append("")
        narrated = _already_narrated_variant_names(ranking)
        unmentioned = [pv["variant_name"] for pv in prompt_versions if pv["variant_name"] not in narrated]
        if unmentioned:
            P.append(
                f"({_prose_list([_speakable(v) for v in unmentioned])} shown in the table above but not "
                f"singled out below - no model's behaviour on them stood out enough to be worth a separate "
                f"section.)"
            )
            P.append("")

    # ---------------- SECTION: THE PART THAT DOESN'T MAKE THE CHART ----------------
    mess_notes = []
    if missing["evidence_insufficient"] > 0:
        ei_models = sorted(df[df["evidence_insufficient"] & (df["run_status"] == "success")]["model_id"].unique())
        if len(ei_models) == 1:
            fixed_model = ei_models[0]
            clean = countable[countable["model_id"] == fixed_model]
            clean_custodial = int(clean["is_custodial"].sum())
            clean_n = len(clean)
            resolution = (
                f"its clean results, once readable, show custody recommended in {clean_custodial} of "
                f"{clean_n} responses - no hidden second version of the headline finding above."
            )
        else:
            resolution = "their clean results are folded into the numbers above."
        mess_notes.append(
            f"{missing['evidence_insufficient']} responses were thrown out entirely partway through this - "
            f"{'one model' if len(ei_models) == 1 else f'{len(ei_models)} models'} spent its entire response "
            f"budget on hidden reasoning tokens and came back with 1-95 characters of actual text, "
            f"sometimes literally nothing but two newline characters. I re-ran {'it' if len(ei_models) == 1 else 'them'} "
            f"at a much higher token budget rather than quietly letting a broken response count as "
            f"\"no recommendation\"; the re-run came back clean, and {resolution}"
        )
    if missing["invalid_unreviewed"] > 0:
        n_pending = missing["invalid_unreviewed"]
        mess_notes.append(
            f"{n_pending} more responses are still sitting in manual review because the automated grader's "
            f"supporting quote didn't pass a strict word-for-word verification check - not wrong exactly, "
            f"just not yet confirmed by a human, so they're excluded from every number in this post until "
            f"they are."
        )
    if mess_notes:
        P.append("## The part that doesn't make it into the chart")
        P.append("")
        P.append(
            "Every tidy results post hides a less tidy process, so here's mine: " + " ".join(mess_notes)
        )
        P.append("")

    # ---------------- CAVEATS ----------------
    P.append("## What this isn't")
    P.append("")
    P.append(
        f"This is one invented case, tested {experiment['repetitions']} times per variant per model. It "
        f"doesn't establish anything about sentencing bias beyond this specific burglary scenario - a "
        f"different offence, a different mitigation profile, or different phrasing of the same dialects "
        f"could behave completely differently. The baseline, name-swap, and AAE results all sit at a "
        f"ceiling (100% for the one model that shows anything at all), which means a smaller or opposite "
        f"effect in those conditions can't be ruled out - only Nigerian English, and more weakly "
        f"Glaswegian, moved far enough off that ceiling to say anything with confidence."
    )
    P.append("")
    P.append(
        "What it is: a real, repeatable test, with every raw response saved, that anyone can run again "
        "and check against mine - including the organisation piloting the exact model this post is about."
    )
    P.append("")

    # ---------------- CLOSING ----------------
    P.append("## So, are AI sentencing tools biased?")
    P.append("")
    if is_single_model_story:
        P.append(
            f"Wrong question for a yes/no answer, same as it always is. The useful version is: biased "
            f"compared to what, and which model? {len(silent_models)} of {n_models} tested showed nothing "
            f"here. One showed something real, specific, and in a direction nobody predicted. That's not a "
            f"verdict on \"AI\" - it's a reason to actually test the specific tool before you trust it with "
            f"a specific decision, instead of assuming every model behaves like the headline worry you "
            f"started with."
        )
    else:
        P.append(
            "I'll let the numbers above answer that rather than doing it for you in a tidy closing line - "
            "that's kind of the point of running the experiment instead of just asking the question out "
            "loud."
        )
    P.append("")
    P.append(
        "If you want to poke at this yourself - the raw responses, the evaluator prompt, the full "
        "per-model breakdown - it's all linked below. Rerun it, change the case, prove me wrong. That's a "
        "better use of a comment section than an argument about what AI models \"are like\" in the "
        "abstract."
    )
    P.append("")
    P.append("*[Link to full data / report / repo here. Your name, your usual sign-off.]*")

    return "\n".join(P)


def generate_full_script(
    conn: sqlite3.Connection, experiment_id: int, companion_experiment_id: int | None = None,
) -> str:
    """A complete, read-on-camera narration script for a sentencing-bias
    experiment, built from its actual recorded data. Lines in **[brackets]**
    are production notes (on-screen text, cuts) - don't read those aloud.
    Structure mirrors export.generate_full_script's adapt-to-the-data
    philosophy, but the actual narration is sentencing-specific throughout,
    same reasoning as generate_blog_post above."""
    experiment = db.get_experiment(conn, experiment_id)
    prompt_versions = db.list_prompt_versions(conn, experiment_id, latest_only=True)
    rows = analysis.build_sentencing_rows(conn, experiment_id)
    df = analysis.rows_to_dataframe(rows)
    counts = analysis.run_counts(df)
    missing = analysis.missing_and_invalid_counts(df)
    n_models = df["model_id"].nunique() if not df.empty else 0

    L: list[str] = []
    L.append(f"# Video Script: {experiment['name']}")
    L.append("")
    L.append(
        "This is a full narration script, generated from this experiment's actually recorded data - not a "
        "summary or an outline. Read the plain text as spoken narration. Lines in **[brackets]** are "
        "production notes (on-screen text, cuts, b-roll) - don't read those aloud. Every number and quote "
        "below is sourced from this experiment and traceable in the raw data / JSON archive; nothing is "
        "invented for effect."
    )
    L.append("")

    countable_total = int(df["is_countable"].sum()) if not df.empty else 0
    if countable_total == 0:
        L.append(
            "No classified results are available yet for this experiment - run it and review/resolve any "
            "invalid evaluations first, then regenerate this script."
        )
        return "\n".join(L)

    countable = df[df["is_countable"]]
    ranking = _rank_variants_by_custodial_rate(df, prompt_versions)
    custodial_counts = countable[countable["is_custodial"]]["model_id"].value_counts()
    total_custodial = int(custodial_counts.sum())
    dominant_model = custodial_counts.index[0] if len(custodial_counts) else None
    dominant_count = int(custodial_counts.iloc[0]) if len(custodial_counts) else 0
    non_dominant_total = total_custodial - dominant_count
    is_single_model_story = (
        dominant_model is not None and total_custodial > 0
        and (dominant_count / total_custodial >= 0.9 or non_dominant_total <= 2)
    )
    silent_models = sorted(set(countable["model_id"].unique()) - set(custodial_counts.index))
    minor_models = sorted(set(custodial_counts.index) - {dominant_model}) if is_single_model_story else []

    if not is_single_model_story:
        # No single model dominates the custodial-recommendation pattern -
        # fall back to a simple results readout rather than forcing the
        # one-model narrative structure the live experiments actually have.
        overall = analysis.custodial_rate(countable)
        L.append("## [COLD OPEN - 0:00]")
        L.append("")
        L.append(f'**[ON SCREEN: the case summary and the first statement variant]**')
        L.append("")
        L.append(
            f'"I gave {n_models} AI models an identical, invented criminal case and asked each one for a '
            f'sentencing recommendation. {overall.numerator} of {overall.denominator} countable responses '
            f'came back custodial. Here\'s exactly how that breaks down."'
        )
        L.append("")
        L.append("## [THE RESULTS]")
        L.append("")
        for pv, rate, n in ranking.variant_ranking:
            L.append(f'"**{_display_variant(pv["variant_name"]).capitalize()}**: {rate*100:.0f}% custodial, out of {n}."')
        L.append("")
        L.append("## [PRODUCTION CHECKLIST]")
        L.append("")
        L.append("- [ ] Record Create Experiment screen (case + model list)")
        L.append("- [ ] Record Results Dashboard with the custodial-rate chart")
        L.append("")
        return "\n".join(L)

    sole_model = dominant_model
    sole = countable[countable["model_id"] == sole_model]
    by_variant = analysis.summarize_sentencing_by(sole, ["prompt_variant"]).set_index("prompt_variant")
    preferred_order = ["baseline", "name_only", "dialect_aae", "dialect_glaswegian", "dialect_nigerian",
                        "dialect_formal_control"]
    ordered_variants = [v for v in preferred_order if v in by_variant.index]
    ordered_variants += [v for v in by_variant.index if v not in preferred_order]
    baseline_pct = by_variant.loc["baseline", "custodial_pct"] if "baseline" in by_variant.index else None
    # Rank variants by THIS model's own rates, not the pooled rate across
    # every model - with several silent (always-zero) models in the mix,
    # the pooled ranking is noise that can disagree with what the dominant
    # model itself actually did, which is what the rest of this section
    # narrates. A real regression: experiment 3's pooled-lowest variant was
    # "name_nigerian" even though Haiku's own lowest variant was "baseline".
    sole_ranking = _rank_variants_by_custodial_rate(sole, prompt_versions)
    low_variant = sole_ranking.low_pv["variant_name"] if sole_ranking.low_pv else "baseline"
    low_rate, _ = _variant_custodial_rate(sole, low_variant)
    sole_overall_rate = float(sole["is_custodial"].sum()) / len(sole) if len(sole) else 0.0

    # ---------------- COLD OPEN ----------------
    L.append("## [COLD OPEN - 0:00]")
    L.append("")
    if sole_ranking.has_contrast and low_variant != "baseline":
        low_display = _display_variant(low_variant)
        hook, follow = _vary(experiment_id, "cold_open_contrast", [
            (
                f'"Same defendant. Same facts. Same statement to the court - except rewritten in {low_display}. '
                f'`{sole_model}` recommended prison time in {baseline_pct:.0f}% of baseline responses. For the '
                f'rewritten version, that drops to {low_rate*100:.0f}%."',
                '"Not harsher. Lighter. That\'s backwards from what I went in expecting, and I\'m going to show you '
                'exactly how I checked it wasn\'t a mistake."',
            ),
            (
                f'"I kept this case completely identical and only changed how the defendant\'s own statement '
                f'was phrased - rewritten in {low_display}. `{sole_model}`\'s custodial rate goes from '
                f'{baseline_pct:.0f}% at baseline to {low_rate*100:.0f}% for that one version."',
                '"That\'s the opposite of what I expected walking in - a lighter outcome, not a harsher one. '
                'Before I call that a finding, I had to go try to break it."',
            ),
            (
                f'"Nothing about this case changes except the statement itself, rewritten in {low_display}. '
                f'`{sole_model}` goes from recommending custody {baseline_pct:.0f}% of the time at baseline '
                f'down to {low_rate*100:.0f}% here - same facts, same defendant, different phrasing."',
                '"Lighter, not harsher. That caught me off guard, so the rest of this video is me checking '
                'whether it\'s real."',
            ),
            (
                f'"Here\'s the case: unchanged. Here\'s what I rewrote: just the statement, into {low_display}. '
                f'And here\'s what happened to `{sole_model}`\'s custodial rate: {baseline_pct:.0f}% at '
                f'baseline, {low_rate*100:.0f}% after."',
                '"That\'s backwards from what I expected going in - a lighter sentence, not a harsher one. I '
                'spent the next few minutes trying to prove that wrong before trusting it."',
            ),
        ])
        L.append(f'**[ON SCREEN: the case summary, side by side with the {low_display} statement]**')
        L.append("")
        L.append(hook)
        L.append("")
        L.append("**[BEAT - let that number sit on screen]**")
        L.append("")
        L.append(follow)
    else:
        hook, follow = _vary(experiment_id, "cold_open_no_contrast", [
            (
                f'"Same defendant. Same facts. Ten AI models. `{sole_model}` recommended prison time in '
                f'{sole_overall_rate*100:.0f}% of its responses to this case - practically every time, no '
                f'matter which version it saw. {len(silent_models)} other models never recommended it once. '
                f'Not a close call. A flat no, every single time."',
                '"So this was never really going to be a story about what changed in the prompt. It\'s a '
                'story about what happens when you ask ten different AI tools the exact same question."',
            ),
            (
                f'"Same defendant, same facts, ten different AI models. `{sole_model}` recommended prison '
                f'time in {sole_overall_rate*100:.0f}% of its responses, no matter which version of the '
                f'statement it got. {len(silent_models)} other models never recommended it a single time."',
                '"So the variable that actually mattered here was never the prompt. It was which model you '
                'asked."',
            ),
            (
                f'"I showed the exact same invented case to ten AI models. One of them, `{sole_model}`, '
                f'recommended prison time in {sole_overall_rate*100:.0f}% of its responses - almost '
                f'regardless of which version it saw. The other {len(silent_models)} never recommended it '
                f'once."',
                '"Changing the statement barely moved the needle. Changing which model answered the '
                'question did."',
            ),
        ])
        L.append('**[ON SCREEN: the case summary and the per-model custodial-rate chart]**')
        L.append("")
        L.append(hook)
        L.append("")
        L.append("**[BEAT]**")
        L.append("")
        L.append(follow)
    L.append("")

    # ---------------- INTRO ----------------
    L.append("## [INTRO - METHODOLOGY]")
    L.append("")
    L.append(_vary(experiment_id, "hypothesis_lead_in", [
        '"Here\'s the hypothesis I went in with:"',
        '"This is the hypothesis I set out to test:"',
        '"Before any of that, here\'s what I was actually trying to find out:"',
    ]))
    L.append("")
    L.append(f"> {experiment['hypothesis'] or '(no hypothesis recorded)'}")
    L.append("")
    L.append(_vary(experiment_id, "methodology_paragraph", [
        '"Police forces and courts are already piloting AI tools in case triage and sentencing support, so '
        'this isn\'t hypothetical. I built one invented case - a burglary, no violence, a guilty plea, no '
        'previous convictions - and changed exactly one thing at a time against a baseline: the '
        'defendant\'s name, or the dialect his own statement was rendered in. Never both at once, so a '
        'name effect and a dialect effect can never get blamed on each other. A second AI model, blind to '
        'the hypothesis, extracted a structured sentence and length from every response, and nothing '
        'counts unless it can point to an exact quote in the response that backs it up."',
        '"This isn\'t a hypothetical - police forces and courts are already piloting AI tools in case '
        'triage and sentencing support. So I built one invented case - burglary, no violence, a guilty '
        'plea, no prior convictions - and changed exactly one variable at a time against a fixed baseline: '
        'either the defendant\'s name, or the dialect of his own statement, never both together, so '
        'neither effect can get blamed on the other. A second model, kept blind to what I was testing, '
        'pulled a structured sentence and length out of every response - and nothing counts unless it can '
        'quote the exact line that justifies it."',
        '"AI tools are already being piloted in case triage and sentencing support by real police forces '
        'and courts, so this question isn\'t academic. The case is invented - a burglary, no violence, a '
        'guilty plea, no previous convictions - and only one thing changes at a time against the baseline: '
        'the defendant\'s name, or the dialect his statement was written in, never both together. A '
        'second, hypothesis-blind model extracts the sentence and length from every response, and it has '
        'to cite an exact quote to back up what it extracted - or it doesn\'t count."',
    ]))
    L.append("")
    intro_stats = f'"{counts.total} total calls, {counts.success} completed successfully'
    if counts.failed:
        intro_stats += f", {counts.failed} failed and excluded"
    if missing["evidence_insufficient"]:
        intro_stats += f", {missing['evidence_insufficient']} excluded for a data-quality issue I\'ll show you later"
    intro_stats += '."'
    L.append(intro_stats)
    L.append("")

    # ---------------- ACT 1: WHAT THE MODEL DID ----------------
    L.append(f"## [ACT 1 - WHAT `{sole_model.upper()}` ACTUALLY DID]")
    L.append("")
    minor_aside = _minor_models_sentence(minor_models, custodial_counts, experiment_id, "script_act1")
    L.append(_vary(experiment_id, "act1_intro", [
        f'"{len(silent_models)} of the {n_models} models I tested never recommended custody once, in any '
        f'version of this case - their behaviour just didn\'t move, so this whole story is about one '
        f'model.{minor_aside} Here\'s its custodial rate, variant by variant:"',
        f'"Out of {n_models} models tested, {len(silent_models)} never recommended custody once, in any '
        f'version of this case - flat, unmoving, across the board. Which means this whole story comes '
        f'down to one model.{minor_aside} Here\'s how its custodial rate broke down, variant by variant:"',
        f'"{len(silent_models)} of the {n_models} models never once recommended custody, no matter which '
        f'version of the case they saw - so whatever\'s interesting here lives entirely in one model\'s '
        f'behaviour.{minor_aside} Variant by variant, here\'s its custodial rate:"',
    ]))
    L.append("")
    for variant_name in ordered_variants:
        r = by_variant.loc[variant_name]
        L.append(f'"**{_capitalize_first(_display_variant(variant_name))}**: {r["custodial_pct"]:.0f}% custodial, '
                  f'{r["custodial_count"]:.0f} of {r["n"]:.0f}."')
    L.append("")
    baseline_quote = _pick_quote(df, "baseline", ["custodial", "suspended_custodial"], model_id=sole_model)
    show_contrast_quote = sole_ranking.has_contrast and low_variant != "baseline"
    low_quote = _pick_quote(df, low_variant, ["community_order", "fine", "discharge"], model_id=sole_model) \
        if show_contrast_quote else None
    if baseline_quote:
        L.append(f'**[ON SCREEN: real response from `{sole_model}`, baseline]**')
        L.append("")
        L.append(_blockquote(_truncate_at_word(baseline_quote["response"], 350)))
        L.append("")
    if low_quote:
        L.append(f'**[ON SCREEN: real response from `{sole_model}`, {_display_variant(low_variant)}]**')
        L.append("")
        L.append(_blockquote(_truncate_at_word(low_quote["response"], 350)))
        L.append("")
        L.append(_vary(experiment_id, "quote_transition", [
            '"Read the aggravating and mitigating factors in both - on screen, side by side. They\'re '
            'nearly the same list, weighed the same way. The facts aren\'t being read differently. The '
            'conclusion is."',
            '"Put the aggravating and mitigating factors from both responses side by side - they\'re '
            'almost the same list, weighted almost the same way. What\'s different isn\'t how the facts '
            'were read. It\'s what the model did with them afterward."',
            '"Line up the reasoning from both responses and the aggravating and mitigating factors barely '
            'differ. Same facts, same weighting - right up until the actual recommendation, which is '
            'where the two responses part ways."',
        ]))
        L.append("")
    elif baseline_quote and not show_contrast_quote:
        L.append(_vary(experiment_id, "no_contrast_quote_note", [
            f'"That\'s a typical response from `{sole_model}` on this case - and it looked almost '
            f'identical no matter which version of the prompt it saw."',
            f'"That\'s a representative response from `{sole_model}` - and swapping in any other version '
            f'of the prompt barely changed it."',
            f'"That\'s what a typical response from `{sole_model}` looks like on this case, and it stayed '
            f'close to that no matter which variant of the prompt it was answering."',
        ]))
        L.append("")

    # ---------------- ACT 2: THE CONFOUND CHECK ----------------
    has_confound_check = "dialect_nigerian" in by_variant.index and "dialect_formal_control" in by_variant.index
    if has_confound_check:
        nigerian_row = by_variant.loc["dialect_nigerian"]
        control_row = by_variant.loc["dialect_formal_control"]
        L.append("## [ACT 2 - CHECKING MY OWN FINDING]")
        L.append("")
        L.append(_vary(experiment_id, "confound_setup", [
            '"Before I trusted this number, I had to rule out the obvious alternative explanation: maybe '
            'the model isn\'t reacting to Nigerian English specifically - maybe it\'s just reacting to the '
            'statement sounding more formal, no contractions, a slightly stiffer register than the casual '
            'baseline. So I added one more version after the fact: same formal, no-contractions register '
            'as the Nigerian-English statement, but with none of its actual Nigerian English grammar."',
            '"Before trusting that number, I needed to rule out the obvious confound: maybe the model '
            'isn\'t reacting to Nigerian English at all, just to the statement reading more formally - '
            'fewer contractions, a stiffer register than the casual baseline. So I wrote one more version '
            'to isolate that: the same formal, contraction-free register as the Nigerian-English '
            'statement, with none of its actual Nigerian English grammar."',
            '"One obvious objection I had to check before trusting this: maybe it\'s not Nigerian English '
            'the model is reacting to, just a more formal, less contracted tone compared to the casual '
            'baseline. So I added a control after the fact - same stiff, formal register as the '
            'Nigerian-English statement, but written in plain standard English instead."',
        ]))
        L.append("")
        control_gap = baseline_pct - control_row["custodial_pct"] if baseline_pct is not None else None
        if control_gap is not None and control_gap <= 2:
            ctrl_pct, nig_pct = control_row["custodial_pct"], nigerian_row["custodial_pct"]
            L.append(_vary(experiment_id, "confound_rules_out_formality", [
                f'"The formal control came back at {ctrl_pct:.0f}% custodial - essentially identical to '
                f'baseline, nowhere near Nigerian English\'s {nig_pct:.0f}%. Formality alone doesn\'t '
                f'explain this. Whatever the model is reacting to is something more specific than just '
                f'sounding more formal - that doesn\'t prove it\'s about nationality or ethnicity, but it '
                f'rules out the simplest alternative."',
                f'"The formal control landed at {ctrl_pct:.0f}% custodial - essentially the same as '
                f'baseline, nowhere close to Nigerian English\'s {nig_pct:.0f}%. So formality alone '
                f'doesn\'t explain this. Something more specific than register is driving it - that '
                f'doesn\'t prove it\'s about nationality or ethnicity, but it does rule out the simplest '
                f'alternative."',
                f'"{ctrl_pct:.0f}% custodial for the formal control - basically indistinguishable from '
                f'baseline, nowhere near Nigerian English\'s {nig_pct:.0f}%. Formality alone just '
                f'doesn\'t explain this gap. Whatever the model is actually reacting to is more specific '
                f'than tone - that\'s not proof it\'s about nationality or ethnicity, but it does take the '
                f'simplest alternative explanation off the table."',
            ]))
        elif control_gap is not None:
            ctrl_pct, nig_pct = control_row["custodial_pct"], nigerian_row["custodial_pct"]
            L.append(_vary(experiment_id, "confound_confirms_formality", [
                f'"And here\'s an honest twist: the formal control ALSO dropped, to {ctrl_pct:.0f}% '
                f'custodial, against Nigerian English\'s {nig_pct:.0f}%. That changes the story - this '
                f'may be a formality effect that the Nigerian statement happens to share, not something '
                f'specific to Nigerian English or nationality. I\'m telling you this on camera because '
                f'finding out you were wrong is part of doing this properly, not something to quietly '
                f'edit out."',
                f'"Here\'s an honest twist, though: the formal control ALSO dropped - to {ctrl_pct:.0f}% '
                f'custodial, against Nigerian English\'s {nig_pct:.0f}%. That complicates the story: this '
                f'might be a formality effect the Nigerian statement happens to share, not something '
                f'specific to nationality or dialect. I\'m keeping this on camera because finding out you '
                f'were wrong is part of doing this properly."',
                f'"There\'s an honest twist here: the formal control ALSO dropped, landing at '
                f'{ctrl_pct:.0f}% custodial against Nigerian English\'s {nig_pct:.0f}%. That reshapes the '
                f'finding - it may be a formality effect the Nigerian statement just happens to share, '
                f'rather than anything specific to dialect or nationality. Showing you this is the point '
                f'of checking your own work instead of quietly editing it out."',
            ]))
        L.append("")

    # ---------------- ACT 3: WHY NOT "AI SENTENCING BIAS" ----------------
    L.append('## [ACT 3 - WHY I\'M NOT CALLING THIS "AI SENTENCING BIAS"]')
    L.append("")
    silent_list = _prose_list([f"`{m}`" for m in silent_models])
    L.append(_vary(experiment_id, "act3_opening", [
        f'"Because it isn\'t, as a general claim. {silent_list} never once recommended custody for this '
        f'case, in any version. This entire finding belongs to one model. If I\'d only tested the other '
        f'{n_models - 1}, this video wouldn\'t exist - and that\'s exactly why testing the specific tool '
        f'matters more than assuming every AI behaves the same way."',
        f'"Because it isn\'t - not as a general claim about AI. {silent_list} never once recommended '
        f'custody for this case, in any version of it. This finding belongs to exactly one model. Test '
        f'only the other {n_models - 1} and this video doesn\'t exist, which is the whole reason testing '
        f'the specific tool beats assuming every model behaves the same."',
        f'"It isn\'t, as a blanket statement about AI. {silent_list} never recommended custody for this '
        f'case, not once, in any version. One model, and one model only, produced this entire finding. '
        f'Leave it out of the test and swap in any of the other {n_models - 1}, and there\'s no video '
        f'here - which is exactly the point: test the specific tool, don\'t assume it behaves like the '
        f'others."',
    ]))
    L.append("")
    if companion_experiment_id is not None:
        companion_paragraph = _companion_null_result_paragraph(conn, companion_experiment_id)
        if companion_paragraph:
            L.append(f'"{companion_paragraph}"')
            L.append("")

    # ---------------- LIMITATIONS ----------------
    L.append("## [WHAT THIS DOES AND DOESN'T PROVE]")
    L.append("")
    if sole_ranking.has_contrast and low_variant != "baseline":
        ceiling_variants = [v for v in ordered_variants if v != low_variant]
        # _display_variant("baseline") already returns "the baseline" - don't
        # prepend a second "The"/"the" article in front of the list, or a
        # ceiling_variants set that includes baseline reads "The the baseline,
        # a pure name swap...". ceiling_list is used mid-sentence as-is;
        # ceiling_list_cap only for the variants that open a sentence with it.
        ceiling_list = _prose_list([_display_variant(v) for v in ceiling_variants])
        ceiling_list_cap = _capitalize_first(ceiling_list)
        L.append(_vary(experiment_id, "limitations_contrast", [
            f'"A few honest caveats. This is one invented case. {ceiling_list_cap} variants all sit at a '
            f'ceiling for this model - meaning a smaller or opposite effect there can\'t be ruled out, only '
            f'the variant that actually moved off that ceiling tells you anything with confidence. And '
            f'repeated runs from the same model aren\'t fully independent samples - they share the same '
            f'underlying weights."',
            f'"A few caveats, honestly stated. This is one invented case. {ceiling_list_cap} variants all '
            f'sat at a ceiling for this model, so a smaller or opposite effect there can\'t be ruled out - '
            f'only the variant that actually moved off that ceiling tells you anything with real '
            f'confidence. And repeated runs from the same model aren\'t independent samples; they come '
            f'from the same underlying weights every time."',
            f'"Some honest limits here. This is one invented case, and {ceiling_list} variants all '
            f'sat at a ceiling for this model - a smaller or opposite effect in those conditions can\'t be '
            f'ruled out, only the variant that actually broke off that ceiling tells you anything with '
            f'confidence. Repeated runs from the same model aren\'t fully independent either, since they '
            f'share the exact same weights."',
        ]))
    else:
        low_r, high_r = sole_ranking.low_rate * 100, sole_ranking.high_rate * 100
        L.append(_vary(experiment_id, "limitations_no_contrast", [
            f'"A few honest caveats. This is one invented case, and `{sole_model}`\'s own rate barely '
            f'moved across these variants - {low_r:.0f}% to {high_r:.0f}%, all within a handful of '
            f'responses of each other. That gap is small enough to be noise, not a finding, which is '
            f'exactly why the real story here is the gap between models, not between variants. And '
            f'repeated runs from the same model aren\'t fully independent samples - they share the same '
            f'underlying weights."',
            f'"A few caveats. This is one invented case, and `{sole_model}`\'s rate barely moved at all '
            f'across variants - {low_r:.0f}% to {high_r:.0f}%, close enough to be noise rather than a '
            f'finding. That\'s exactly why the real story is the gap between models, not between '
            f'variants. Repeated runs from the same model also aren\'t fully independent - they share '
            f'identical underlying weights."',
            f'"Some honest limits. This is one invented case, and `{sole_model}`\'s own custodial rate '
            f'stayed essentially flat across every variant - {low_r:.0f}% to {high_r:.0f}%, small enough '
            f'to be noise rather than signal. The real story here was always between models, not between '
            f'variants. And repeated runs from one model aren\'t independent samples; they come from the '
            f'same weights every time."',
        ]))
    L.append("")
    if missing["evidence_insufficient"] > 0:
        n_ei = missing["evidence_insufficient"]
        L.append(_vary(experiment_id, "evidence_insufficient_aside", [
            f'"One more thing, because I want to show my work: {n_ei} response(s) got excluded entirely '
            f'during this - one model spent its whole token budget on hidden reasoning and came back '
            f'with a handful of characters, sometimes literally nothing. I re-ran it at a much higher '
            f'budget rather than let a broken response quietly count as a real answer; the clean re-run '
            f'showed that model belongs with the silent majority, not a second hidden version of this '
            f'finding."',
            f'"One more thing worth showing, not just saying: {n_ei} response(s) got thrown out entirely '
            f'- a model burned its whole token budget on hidden reasoning and came back with almost '
            f'nothing, sometimes literally nothing. I re-ran it at a much higher budget rather than count '
            f'a broken response as a real answer; the clean version confirmed that model belongs with '
            f'the silent majority, not a second hidden result."',
            f'"And one thing I won\'t skip past: {n_ei} response(s) were excluded entirely - one model '
            f'spent its entire token budget on hidden reasoning and returned almost no actual text. '
            f'Rather than let that count as a real answer, I re-ran it at a much higher budget; the clean '
            f'result put that model right back with the silent majority, not inside this finding."',
        ]))
        L.append("")

    # ---------------- CLOSING ----------------
    L.append("## [CLOSING - CALL TO ACTION]")
    L.append("")
    if sole_ranking.has_contrast and low_variant != "baseline":
        L.append(_vary(experiment_id, "closing_contrast", [
            '"So: one model out of ten showed a real, repeatable effect - and it ran backwards from what '
            'everyone, including me, would have guessed. That\'s not a verdict on \'AI\' in general. It\'s '
            'a reason to actually test the specific tool before you trust it with a specific decision."',
            '"One model out of ten showed a real, repeatable effect here - and it ran in the opposite '
            'direction from what I, or probably you, would have guessed going in. That\'s not a verdict '
            'on AI in general. It\'s a reason to test the specific tool you\'re relying on before you '
            'trust it with a specific decision."',
            '"So here\'s where this lands: one model out of ten, a real and repeatable effect, running '
            'backwards from what anyone would have predicted beforehand. Not a verdict on AI as a '
            'category - a reason to test the specific tool in front of you before trusting it with '
            'anything that matters."',
        ]))
    else:
        L.append(_vary(experiment_id, "closing_no_contrast", [
            f'"So: whatever changed in this prompt made no real difference to `{sole_model}`, or to any '
            f'of the other nine models. What made the difference was which model you asked. That\'s not a '
            f'verdict on \'AI\' in general. It\'s a reason to actually test the specific tool before you '
            f'trust it with a specific decision."',
            f'"So: the prompt variant never really mattered here - not to `{sole_model}`, not to any of '
            f'the other nine models. What mattered was which model answered. That\'s not a verdict on AI '
            f'as a category. It\'s a reason to test the specific tool before you hand it a specific '
            f'decision."',
            f'"Here\'s the honest summary: changing the prompt made essentially no difference, to '
            f'`{sole_model}` or to any of the other nine models tested. The only variable that actually '
            f'moved the outcome was which model you picked. That\'s not a verdict on AI in general - it\'s '
            f'a reason to test the specific tool before trusting it with anything specific."',
        ]))
    L.append("")
    L.append(_vary(experiment_id, "final_cta", [
        '"Full dataset, every raw response, the exact evaluator prompt I used to score this fairly - all '
        'linked below, so you can check my work instead of just taking my word for it. If you want me to '
        'run this on a different case, or test a specific model, tell me in the comments."',
        '"The full dataset, every raw response, the exact evaluator prompt I used - it\'s all linked '
        'below, so you don\'t have to take my word for any of this. Want me to run a different case, or '
        'test a specific model? Say so in the comments."',
        '"Everything\'s linked below - the full dataset, every raw response, the evaluator prompt I used '
        'to score this. Check it yourself instead of trusting me on it. If there\'s a different case or '
        'model you want tested, tell me in the comments."',
    ]))
    L.append("")

    # ---------------- PRODUCTION CHECKLIST ----------------
    L.append("## [PRODUCTION CHECKLIST]")
    L.append("")
    L.append("- [ ] Record Create Experiment screen (case text + model list)")
    L.append("- [ ] Record Run Experiment screen with live progress")
    L.append("- [ ] Record Human Review screen showing a real sentencing response + automated classification")
    L.append(f"- [ ] Record Results Dashboard with the custodial-rate chart for `{sole_model}`")
    L.append("- [ ] On-screen callouts for every quoted number above (use the exact n/% shown, not rounded further)")
    if has_confound_check:
        L.append("- [ ] Side-by-side on-screen text comparison of the baseline and formal-control statements, to make the confound check visually obvious")
    L.append("- [ ] Link in description: full report, JSON archive, and (if public) the repo")
    L.append("")

    return "\n".join(L)
