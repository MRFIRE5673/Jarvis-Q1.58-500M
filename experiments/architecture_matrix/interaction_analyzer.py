# experiments/architecture_matrix/interaction_analyzer.py
"""
INTERACTION EFFECT MATRIX ANALYZER
===================================
Computes empirical interaction coefficients across memory mechanism pairs:
  Effect(A) = CE(A) - CE(baseline)
  Effect(B) = CE(B) - CE(baseline)
  Expected(A+B) = Effect(A) + Effect(B)
  Actual(A+B) = CE(A+B) - CE(baseline)
  Interaction(A, B) = Actual(A+B) - Expected(A+B)

Synergy Classification:
- Interaction < -0.01: SYNERGISTIC (Cooperative)
- |Interaction| <= 0.01: ADDITIVE (Independent)
- Interaction > +0.01: ANTAGONISTIC (Interfering / Redundant)
"""

import os
import json
import argparse

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(WORKSPACE_ROOT, "experiments", "research_database.json")
OUTPUT_MD = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix", "interaction_matrix.md")


def analyze_interactions():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    exps = db.get("experiments", {})
    baseline_ce = exps.get("baseline", {}).get("final_ce", 3.2858)

    # Component aliases
    components = {
        "A": "Adaptive Decay",
        "B": "Write Gate",
        "C": "Erase Gate",
        "D": "Gated Read",
        "E": "Local Buffer",
    }

    # Extract single component CE
    # Look for fact_<X> or specific known runs
    single_map = {}
    for c in ["A", "B", "C", "D", "E"]:
        fact_key = f"fact_{c}"
        if fact_key in exps:
            single_map[c] = exps[fact_key]["final_ce"]
        elif c == "A" and "exp_adaptive_decay" in exps:
            single_map[c] = exps["exp_adaptive_decay"]["final_ce"]
        elif c == "B" and "exp_gated_write" in exps:
            single_map[c] = exps["exp_gated_write"]["final_ce"]
        elif c == "C" and "exp_erase_gate" in exps:
            single_map[c] = exps["exp_erase_gate"]["final_ce"]
        elif c == "D" and "exp_gated_read" in exps:
            single_map[c] = exps["exp_gated_read"]["final_ce"]
        elif c == "E" and "exp_sliding_buffer" in exps:
            single_map[c] = exps["exp_sliding_buffer"]["final_ce"]

    # Pairwise combinations
    pairs = [
        ("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"),
        ("B", "C"), ("B", "D"), ("B", "E"),
        ("C", "D"), ("C", "E"),
        ("D", "E"),
    ]

    results = []
    for c1, c2 in pairs:
        pair_code = f"{c1}+{c2}"
        fact_key = f"fact_{c1}+{c2}"
        
        # Check alternate names
        pair_ce = None
        if fact_key in exps:
            pair_ce = exps[fact_key]["final_ce"]
        elif pair_code == "B+C" and "exp_write_erase_gate" in exps:
            pair_ce = exps["exp_write_erase_gate"]["final_ce"]
        elif pair_code == "B+E" and "v2_ablate_gelu" in exps:
            # Buffer + Write
            pass

        if pair_ce is not None and c1 in single_map and c2 in single_map:
            eff1 = single_map[c1] - baseline_ce
            eff2 = single_map[c2] - baseline_ce
            expected_delta = eff1 + eff2
            actual_delta = pair_ce - baseline_ce
            interaction = actual_delta - expected_delta

            if interaction < -0.01:
                verdict = "SYNERGISTIC"
            elif interaction > 0.01:
                verdict = "ANTAGONISTIC"
            else:
                verdict = "ADDITIVE"

            results.append({
                "pair": pair_code,
                "name1": components[c1],
                "name2": components[c2],
                "ce1": single_map[c1],
                "ce2": single_map[c2],
                "pair_ce": pair_ce,
                "eff1": eff1,
                "eff2": eff2,
                "expected_delta": expected_delta,
                "actual_delta": actual_delta,
                "interaction": interaction,
                "verdict": verdict,
            })

    # Write report
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# PAIRWISE MEMORY INTERACTION MATRIX\n\n")
        f.write(f"**Baseline CE**: {baseline_ce:.4f}\n\n")
        f.write("| Pair | Mechanism 1 | Mechanism 2 | Single 1 CE | Single 2 CE | Actual Pair CE | Expected $\\Delta$ | Actual $\\Delta$ | Interaction | Classification |\n")
        f.write("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for r in results:
            f.write(f"| `{r['pair']}` | {r['name1']} | {r['name2']} | {r['ce1']:.4f} | {r['ce2']:.4f} | **{r['pair_ce']:.4f}** | {r['expected_delta']:+.4f} | {r['actual_delta']:+.4f} | {r['interaction']:+.4f} | **{r['verdict']}** |\n")

    print(f"Generated interaction analysis for {len(results)} pairs at: {OUTPUT_MD}")
    return results


if __name__ == "__main__":
    analyze_interactions()
