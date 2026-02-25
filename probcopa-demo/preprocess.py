#!/usr/bin/env python3
"""
Preprocess ProbCOPA data into a compact JSON for the interactive demo.
Reads raw JSONL files from the probabilistic-reasoning-clean repo and outputs data.json.

Usage:
    python preprocess.py --data-dir ../../probabilistic-reasoning-clean
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


MODELS = [
    "gpt-5",
    "claude-sonnet-4.5",
    "DeepSeek-R1",
    "gemini-3-pro-preview",
    "Kimi-K2-Thinking",
    "Qwen3-235B-Thinking",
    "GLM-4.6",
    "grok-4.1-fast",
]

MODEL_DISPLAY_NAMES = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "DeepSeek-R1": "DeepSeek-R1",
    "gemini-3-pro-preview": "Gemini-3",
    "Kimi-K2-Thinking": "Kimi-K2",
    "Qwen3-235B-Thinking": "Qwen3-235B",
    "GLM-4.6": "GLM-4.6",
    "grok-4.1-fast": "Grok-4.1",
}

TEMPERATURES = [0.4, 0.8, 1.2, 1.6, 2.0]

TEMP_MODELS = [
    "DeepSeek-R1", "GLM-4.6", "Kimi-K2-Thinking",
    "Qwen3-235B-Thinking", "gemini-3-pro-preview", "grok-4.1-fast",
]

REASONING_EFFORT_FILES = {
    "DeepSeek-R1": ["low", "medium", "high"],
    "GLM-4.6": ["low", "medium", "high"],
    "Kimi-K2-Thinking": ["low", "medium", "high"],
    "Qwen3-235B-Thinking": ["low", "medium", "high"],
    "claude-opus-4.6": ["low", "medium", "high"],
    "gpt-5": ["low", "medium", "high"],
}

THINKING_BUDGET_FILES = {
    "claude-sonnet-4.5": [512, 2048, 4096],
    "gemini-3-pro-preview": [512, 2048, 4096],
}


def read_jsonl(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def estimate_differential_entropy(values, sigma=0.1, random_seed=3535):
    np.random.seed(random_seed)
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return None
    noise = np.random.normal(0, sigma, len(arr))
    return float(stats.differential_entropy(arr + noise))


def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def compute_wasserstein(model_responses, human_responses):
    m = [v for v in model_responses if v is not None]
    h = [v for v in human_responses if v is not None]
    if len(m) < 2 or len(h) < 2:
        return None
    return float(stats.wasserstein_distance(m, h))


def extract_canary_hex(entries):
    for e in entries:
        if e.get("type") == "canary":
            return e["canary_hex"]
    return None


def insert_canaries_into_list(lst, canary_hex, num_canaries=10, seed=42):
    """Insert canary objects at random positions in a list."""
    rng = random.Random(seed)
    canary = {"type": "canary", "canary_hex": canary_hex}
    n = min(num_canaries, len(lst) + 1)
    positions = sorted(rng.sample(range(len(lst) + 1), n))
    result = []
    lst_idx = 0
    canary_idx = 0
    for pos in range(len(lst) + 1):
        if canary_idx < len(positions) and positions[canary_idx] == pos:
            result.append(canary)
            canary_idx += 1
        if lst_idx < len(lst):
            result.append(lst[lst_idx])
            lst_idx += 1
    return result


def insert_canaries_into_response_arrays(responses_map, canary_hex, seed_base=100):
    """Insert canary values into some of the per-UID response arrays."""
    rng = random.Random(seed_base)
    uids = list(responses_map.keys())
    n_to_canary = min(20, len(uids))
    chosen = rng.sample(uids, n_to_canary)
    canary = {"type": "canary", "canary_hex": canary_hex}
    for uid in chosen:
        arr = responses_map[uid]
        pos = rng.randint(0, len(arr))
        arr.insert(pos, canary)
    return responses_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Path to probabilistic-reasoning-clean repo")
    parser.add_argument("--output", default="data.json",
                        help="Output JSON file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = data_dir / "results"
    datasets_dir = data_dir / "datasets"

    # ---- 1. Items ----
    print("Reading items...")
    items_raw = read_jsonl(datasets_dir / "probcopa_items.jsonl")
    items = []
    for it in items_raw:
        items.append({
            "UID": it["UID"],
            "premise": it["premise"],
            "hypothesis": it["hypothesis"],
            "hard_label": it["hard_label"],
        })
    uid_set = {it["UID"] for it in items}
    print(f"  {len(items)} items")

    # ---- 2. Human data (CANARY version) ----
    print("Reading human data (CANARY)...")
    human_raw = read_jsonl(results_dir / "probcopa_human_results_annotated_CANARY.jsonl")

    canary_hex = extract_canary_hex(human_raw)
    print(f"  Canary hex: {canary_hex[:16]}...")

    human_entries = [e for e in human_raw if e.get("type") != "canary"]

    human_responses = defaultdict(list)
    human_aggregates = {}
    for e in human_entries:
        uid = e["UID"]
        resp = safe_float(e.get("response"))
        if resp is not None:
            human_responses[uid].append(resp)
        if uid not in human_aggregates:
            human_aggregates[uid] = {
                "median": e.get("median_response"),
                "mean": round(e.get("mean_response", 0), 2) if e.get("mean_response") is not None else None,
                "std": round(e.get("std_response", 0), 2) if e.get("std_response") is not None else None,
                "diff_entropy": round(e.get("diff_entropy_response", 0), 4) if e.get("diff_entropy_response") is not None else None,
            }
    print(f"  {len(human_responses)} UIDs, {sum(len(v) for v in human_responses.values())} total responses")

    # ---- 3. Model data ----
    print("Reading model data...")
    model_responses = {}
    model_aggregates = {}

    for model in MODELS:
        fpath = results_dir / f"probcopa_{model}.jsonl"
        if not fpath.exists():
            print(f"  SKIP {model} (file not found)")
            continue

        raw = read_jsonl(fpath)
        entries = [e for e in raw if e.get("type") != "canary"]

        per_uid_responses = defaultdict(list)
        per_uid_reasoning_tokens = defaultdict(list)
        for e in entries:
            uid = e["UID"]
            val = safe_float(e.get("answer"))
            if val is not None:
                per_uid_responses[uid].append(val)
            rtc = e.get("reasoning_token_count")
            if rtc is not None:
                per_uid_reasoning_tokens[uid].append(rtc)

        resp_map = {}
        agg_map = {}
        for uid in uid_set:
            resps = per_uid_responses.get(uid, [])
            resp_map[uid] = resps
            valid = [v for v in resps if v is not None]
            h_resps = human_responses.get(uid, [])
            rtokens = per_uid_reasoning_tokens.get(uid, [])

            agg_map[uid] = {
                "median": float(np.median(valid)) if valid else None,
                "mean": round(float(np.mean(valid)), 2) if valid else None,
                "std": round(float(np.std(valid)), 2) if valid else None,
                "diff_entropy": estimate_differential_entropy(valid) if len(valid) >= 2 else None,
                "wasserstein": compute_wasserstein(valid, h_resps),
                "mean_reasoning_tokens": round(float(np.mean(rtokens)), 1) if rtokens else None,
            }
            if agg_map[uid]["diff_entropy"] is not None:
                agg_map[uid]["diff_entropy"] = round(agg_map[uid]["diff_entropy"], 4)
            if agg_map[uid]["wasserstein"] is not None:
                agg_map[uid]["wasserstein"] = round(agg_map[uid]["wasserstein"], 2)

        model_responses[model] = resp_map
        model_aggregates[model] = agg_map
        print(f"  {model}: {len(per_uid_responses)} UIDs")

    # ---- 4. Temperature ablation ----
    print("Reading temperature ablation data...")
    temp_ablation = []
    temp_dir = results_dir / "temperature_experiments"
    for model in TEMP_MODELS:
        for temp in TEMPERATURES:
            fpath = temp_dir / f"{model}_temperature_{temp:.1f}.jsonl"
            if not fpath.exists():
                continue
            raw = read_jsonl(fpath)
            entries = [e for e in raw if e.get("type") != "canary"]

            per_uid = defaultdict(list)
            for e in entries:
                val = safe_float(e.get("answer"))
                if val is not None:
                    per_uid[e["UID"]].append(val)

            entropies = []
            wassersteins = []
            for uid in uid_set:
                resps = per_uid.get(uid, [])
                if len(resps) >= 2:
                    ent = estimate_differential_entropy(resps)
                    if ent is not None:
                        entropies.append(ent)
                    h = human_responses.get(uid, [])
                    w = compute_wasserstein(resps, h)
                    if w is not None:
                        wassersteins.append(w)

            temp_ablation.append({
                "model": model,
                "temperature": temp,
                "mean_entropy": round(float(np.mean(entropies)), 4) if entropies else None,
                "median_entropy": round(float(np.median(entropies)), 4) if entropies else None,
                "mean_wasserstein": round(float(np.mean(wassersteins)), 2) if wassersteins else None,
                "median_wasserstein": round(float(np.median(wassersteins)), 2) if wassersteins else None,
                "n_items": len(entropies),
            })

    print(f"  {len(temp_ablation)} temperature conditions")

    # ---- 5. Reasoning effort ablation ----
    print("Reading reasoning effort ablation data...")
    effort_ablation = []
    effort_dir = results_dir / "reasoning_effort_experiments"

    for model, levels in REASONING_EFFORT_FILES.items():
        for level in levels:
            fpath = effort_dir / f"{model}_reasoning_effort_{level}.jsonl"
            if not fpath.exists():
                continue
            raw = read_jsonl(fpath)
            entries = [e for e in raw if e.get("type") != "canary"]

            per_uid = defaultdict(list)
            for e in entries:
                val = safe_float(e.get("answer"))
                if val is not None:
                    per_uid[e["UID"]].append(val)

            entropies = []
            wassersteins = []
            for uid in uid_set:
                resps = per_uid.get(uid, [])
                if len(resps) >= 2:
                    ent = estimate_differential_entropy(resps)
                    if ent is not None:
                        entropies.append(ent)
                    h = human_responses.get(uid, [])
                    w = compute_wasserstein(resps, h)
                    if w is not None:
                        wassersteins.append(w)

            effort_ablation.append({
                "model": model,
                "condition": f"effort_{level}",
                "mean_entropy": round(float(np.mean(entropies)), 4) if entropies else None,
                "median_entropy": round(float(np.median(entropies)), 4) if entropies else None,
                "mean_wasserstein": round(float(np.mean(wassersteins)), 2) if wassersteins else None,
                "median_wasserstein": round(float(np.median(wassersteins)), 2) if wassersteins else None,
                "n_items": len(entropies),
            })

    for model, budgets in THINKING_BUDGET_FILES.items():
        for budget in budgets:
            fpath = effort_dir / f"{model}_thinking_budget_{budget}.jsonl"
            if not fpath.exists():
                continue
            raw = read_jsonl(fpath)
            entries = [e for e in raw if e.get("type") != "canary"]

            per_uid = defaultdict(list)
            for e in entries:
                val = safe_float(e.get("answer"))
                if val is not None:
                    per_uid[e["UID"]].append(val)

            entropies = []
            wassersteins = []
            for uid in uid_set:
                resps = per_uid.get(uid, [])
                if len(resps) >= 2:
                    ent = estimate_differential_entropy(resps)
                    if ent is not None:
                        entropies.append(ent)
                    h = human_responses.get(uid, [])
                    w = compute_wasserstein(resps, h)
                    if w is not None:
                        wassersteins.append(w)

            effort_ablation.append({
                "model": model,
                "condition": f"budget_{budget}",
                "mean_entropy": round(float(np.mean(entropies)), 4) if entropies else None,
                "median_entropy": round(float(np.median(entropies)), 4) if entropies else None,
                "mean_wasserstein": round(float(np.mean(wassersteins)), 2) if wassersteins else None,
                "median_wasserstein": round(float(np.median(wassersteins)), 2) if wassersteins else None,
                "n_items": len(entropies),
            })

    print(f"  {len(effort_ablation)} reasoning effort/budget conditions")

    # ---- 6. Human validation baseline (hold-out participant group on random 30-item sample) ----
    print("Reading human validation baseline...")
    val_path = results_dir / "probcopa_random_sample_validation_round_human_results_CANARY.jsonl"
    val_raw = read_jsonl(val_path)
    val_entries = [e for e in val_raw if e.get("type") != "canary"]

    val_responses = defaultdict(list)
    for e in val_entries:
        resp = safe_float(e.get("response"))
        if resp is not None:
            val_responses[e["UID"]].append(resp)

    human_validation = {}
    val_entropies = []
    val_wassersteins = []
    for uid, resps in val_responses.items():
        ent = estimate_differential_entropy(resps) if len(resps) >= 2 else None
        h_resps = human_responses.get(uid, [])
        wass = compute_wasserstein(resps, h_resps)
        med = float(np.median(resps)) if resps else None
        human_validation[str(uid)] = {
            "diff_entropy": round(ent, 4) if ent is not None else None,
            "wasserstein": round(wass, 2) if wass is not None else None,
            "median": med,
            "human_median": human_aggregates.get(uid, {}).get("median"),
            "human_diff_entropy": human_aggregates.get(uid, {}).get("diff_entropy"),
        }
        if ent is not None:
            val_entropies.append(ent)
        if wass is not None:
            val_wassersteins.append(wass)

    human_validation_summary = {
        "mean_entropy": round(float(np.mean(val_entropies)), 4) if val_entropies else None,
        "median_entropy": round(float(np.median(val_entropies)), 4) if val_entropies else None,
        "mean_wasserstein": round(float(np.mean(val_wassersteins)), 2) if val_wassersteins else None,
        "median_wasserstein": round(float(np.median(val_wassersteins)), 2) if val_wassersteins else None,
    }
    print(f"  {len(val_responses)} UIDs, summary: {human_validation_summary}")

    # ---- 7. Persona prompt ablation ----
    print("Reading persona prompt ablation data...")
    persona_ablation = []
    persona_dir = results_dir / "persona_prompt_experiments"
    persona_types = ["demographic", "psychological"]
    for model in MODELS:
        for ptype in persona_types:
            fpath = persona_dir / f"{model}_structured_personas_{ptype}.jsonl"
            if not fpath.exists():
                continue
            raw = read_jsonl(fpath)
            entries = [e for e in raw if e.get("type") != "canary"]

            per_uid = defaultdict(list)
            for e in entries:
                val = safe_float(e.get("answer"))
                if val is not None:
                    per_uid[e["UID"]].append(val)

            entropies = []
            wassersteins = []
            for uid in uid_set:
                resps = per_uid.get(uid, [])
                if len(resps) >= 2:
                    ent = estimate_differential_entropy(resps)
                    if ent is not None:
                        entropies.append(ent)
                    h = human_responses.get(uid, [])
                    w = compute_wasserstein(resps, h)
                    if w is not None:
                        wassersteins.append(w)

            persona_ablation.append({
                "model": model,
                "persona_type": ptype,
                "mean_entropy": round(float(np.mean(entropies)), 4) if entropies else None,
                "median_entropy": round(float(np.median(entropies)), 4) if entropies else None,
                "mean_wasserstein": round(float(np.mean(wassersteins)), 2) if wassersteins else None,
                "median_wasserstein": round(float(np.median(wassersteins)), 2) if wassersteins else None,
                "n_items": len(entropies),
            })

    print(f"  {len(persona_ablation)} persona conditions")

    # ---- 8. Also compute main model default aggregates for ablation comparison ----
    main_model_agg_summary = {}
    for model in MODELS:
        if model not in model_aggregates:
            continue
        agg = model_aggregates[model]
        entropies = [v["diff_entropy"] for v in agg.values() if v["diff_entropy"] is not None]
        wassersteins = [v["wasserstein"] for v in agg.values() if v["wasserstein"] is not None]
        main_model_agg_summary[model] = {
            "mean_entropy": round(float(np.mean(entropies)), 4) if entropies else None,
            "median_entropy": round(float(np.median(entropies)), 4) if entropies else None,
            "mean_wasserstein": round(float(np.mean(wassersteins)), 2) if wassersteins else None,
            "median_wasserstein": round(float(np.median(wassersteins)), 2) if wassersteins else None,
        }

    # ---- 7. Assemble and insert canaries ----
    print("Assembling output...")

    # Convert defaultdicts and numpy types
    human_responses_out = {str(k): [round(v, 1) for v in vals] for k, vals in human_responses.items()}
    human_aggregates_out = {str(k): v for k, v in human_aggregates.items()}
    model_responses_out = {}
    model_aggregates_out = {}
    for model in MODELS:
        if model not in model_responses:
            continue
        model_responses_out[model] = {
            str(k): [round(v, 1) if v is not None else None for v in vals]
            for k, vals in model_responses[model].items()
        }
        model_aggregates_out[model] = {str(k): v for k, v in model_aggregates[model].items()}

    # Insert canary strings
    if canary_hex:
        print(f"Inserting canary strings (hex: {canary_hex[:16]}...)")
        items = insert_canaries_into_list(items, canary_hex, num_canaries=10, seed=42)
        human_responses_out = insert_canaries_into_response_arrays(
            human_responses_out, canary_hex, seed_base=200
        )

    output = {
        "metadata": {
            "n_items": 210,
            "models": MODELS,
            "model_display_names": MODEL_DISPLAY_NAMES,
            "description": "ProbCOPA: Probabilistic Inferences in Humans and LLMs",
        },
        "items": items,
        "human_responses": human_responses_out,
        "human_aggregates": human_aggregates_out,
        "model_responses": model_responses_out,
        "model_aggregates": model_aggregates_out,
        "human_validation": human_validation,
        "human_validation_summary": human_validation_summary,
        "temperature_ablation": temp_ablation,
        "reasoning_effort_ablation": effort_ablation,
        "persona_ablation": persona_ablation,
        "main_model_summary": main_model_agg_summary,
    }

    out_path = Path(args.output)

    def sanitize(obj):
        """Recursively replace NaN/Infinity with None for valid JSON."""
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    output = sanitize(output)

    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
