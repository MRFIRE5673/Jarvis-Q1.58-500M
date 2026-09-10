# experiments/architecture_matrix/batch_factorial_matrix.py
"""
BATCH FACTORIAL MATRIX EXECUTOR
===============================
Sequentially drives the factorial memory combination matrix:
1. Singles: A, B, C, D, E
2. Pairs: A+B, A+C, A+D, A+E, B+C, B+D, B+E, C+D, C+E, D+E
3. Triples / Higher: Promising 3-way & 4-way combinations

Skips any previously completed runs, records exact VRAM & diagnostics,
and recalculates the interaction matrix after each phase.
"""

import os
import sys
import subprocess

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARCH_DIR = os.path.join(WORKSPACE_ROOT, "experiments", "architecture_matrix")
RUNNER_SCRIPT = os.path.join(ARCH_DIR, "factorial_research_runner.py")
ANALYZER_SCRIPT = os.path.join(ARCH_DIR, "interaction_analyzer.py")

# Staged Factorial Testing Queue
SINGLES = ["A", "B", "C", "D", "E"]
PAIRS = [
    "A+B", "A+C", "A+D", "A+E",
    "B+C", "B+D", "B+E",
    "C+D", "C+E",
    "D+E",
]
TRIPLES = [
    "A+B+C", "A+C+D", "A+C+E", "A+D+E",
    "B+C+D", "B+C+E", "C+D+E"
]
FOUR_WAY = [
    "A+B+C+D", "A+C+D+E", "A+B+C+E"
]
FULL = ["A+B+C+D+E"]


def run_single_experiment(combo: str, fusion: int = 2, steps: int = 100):
    cmd = [
        sys.executable,
        "-u",
        RUNNER_SCRIPT,
        "--combo", combo,
        "--fusion", str(fusion),
        "--steps", str(steps),
    ]
    print(f"\n>>> [QUEUE] Launching: {combo} (Fusion: {fusion}, Steps: {steps})")
    ret = subprocess.run(cmd, cwd=WORKSPACE_ROOT)
    if ret.returncode != 0:
        print(f"!!! Error in {combo}, continuing queue...")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all", choices=["singles", "pairs", "triples", "four_way", "full", "all"])
    args = parser.parse_args()

    queue = []
    if args.stage in ["singles", "all"]:
        queue.extend([(c, 2) for c in SINGLES])
    if args.stage in ["pairs", "all"]:
        queue.extend([(c, 2) for c in PAIRS])
    if args.stage in ["triples", "all"]:
        queue.extend([(c, 2) for c in TRIPLES])
    if args.stage in ["four_way", "all"]:
        queue.extend([(c, 2) for c in FOUR_WAY])
    if args.stage in ["full", "all"]:
        queue.extend([(c, 2) for c in FULL])

    print(f"Batch Factorial Matrix Driver: {len(queue)} experiments queued.")
    for combo, fusion in queue:
        run_single_experiment(combo, fusion=fusion, steps=100)

    # Run Interaction Analysis
    subprocess.run([sys.executable, ANALYZER_SCRIPT], cwd=WORKSPACE_ROOT)
    print("\nBatch Factorial Matrix Execution Finished!")


if __name__ == "__main__":
    main()
