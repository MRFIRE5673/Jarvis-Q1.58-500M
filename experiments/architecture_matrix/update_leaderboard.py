import json
import os

db_path = r"e:\Jarvis-Q1.58-500M\experiments\research_database.json"
md_path = r"e:\Jarvis-Q1.58-500M\experiments\leaderboard.md"

known = [
    {
        "experiment_id": "baseline",
        "category": "Frozen Baseline",
        "initial_ce": 3.2858,
        "final_ce": 3.2858,
        "ce_delta": 0.0,
        "final_ppl": 26.73,
        "tok_s": 890.0,
        "needle_rank_64": 4053.0,
        "checkpoint": r"jarvis_engine\ckpt_step_0004209.pt"
    },
    {
        "experiment_id": "exp_aux_free_bias",
        "category": "MoE Routing",
        "initial_ce": 3.2857,
        "final_ce": 3.2985,
        "ce_delta": 0.0128,
        "final_ppl": 27.07,
        "tok_s": 604.0,
        "needle_rank_64": 3042.0,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_aux_free_bias.pt"
    },
    {
        "experiment_id": "exp_squared_relu",
        "category": "FFN Activations",
        "initial_ce": 3.4495,
        "final_ce": 3.3049,
        "ce_delta": -0.1446,
        "final_ppl": 27.24,
        "tok_s": 568.0,
        "needle_rank_64": 3108.0,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_squared_relu.pt"
    },
    {
        "experiment_id": "exp_delta_memory",
        "category": "Associative Memory",
        "initial_ce": 9.4328,
        "final_ce": 3.3231,
        "ce_delta": -6.1097,
        "final_ppl": 27.75,
        "tok_s": 392.0,
        "needle_rank_64": 7774.6,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_delta_memory.pt"
    },
    {
        "experiment_id": "exp_swiglu",
        "category": "FFN Activations",
        "initial_ce": 4.5371,
        "final_ce": 3.3936,
        "ce_delta": -1.1435,
        "final_ppl": 29.78,
        "tok_s": 257.0,
        "needle_rank_64": 3180.0,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_swiglu.pt"
    },
    {
        "experiment_id": "exp_deepseek_shared",
        "category": "MoE Routing",
        "initial_ce": 4.5371,
        "final_ce": 3.3992,
        "ce_delta": -1.1379,
        "final_ppl": 29.94,
        "tok_s": 546.0,
        "needle_rank_64": 3150.0,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_deepseek_shared.pt"
    },
    {
        "experiment_id": "exp_muon_hybrid",
        "category": "Optimizer Dynamics",
        "initial_ce": 3.2858,
        "final_ce": 3.4434,
        "ce_delta": 0.1576,
        "final_ppl": 31.29,
        "tok_s": 540.0,
        "needle_rank_64": 3200.0,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_muon_hybrid.pt"
    },
    {
        "experiment_id": "exp_sliding_buffer",
        "category": "Attention & Recurrence",
        "initial_ce": 4.7329,
        "final_ce": 3.6792,
        "ce_delta": -1.0537,
        "final_ppl": 39.61,
        "tok_s": 863.0,
        "needle_rank_64": 2016.0,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_sliding_buffer.pt"
    },
    {
        "experiment_id": "exp_gated_write",
        "category": "Attention & Recurrence",
        "initial_ce": 5.2263,
        "final_ce": 3.7341,
        "ce_delta": -1.4922,
        "final_ppl": 41.85,
        "tok_s": 950.0,
        "needle_rank_64": 2664.0,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_gated_write.pt"
    },
    {
        "experiment_id": "exp_multi_timescale",
        "category": "Attention & Recurrence",
        "initial_ce": 5.4192,
        "final_ce": 3.8256,
        "ce_delta": -1.5935,
        "final_ppl": 45.86,
        "tok_s": 870.0,
        "needle_rank_64": 1831.0,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_multi_timescale.pt"
    },
    {
        "experiment_id": "exp_buffer_w8",
        "category": "Memory Buffer",
        "initial_ce": 4.7336,
        "final_ce": 3.6951,
        "ce_delta": -1.0385,
        "final_ppl": 40.25,
        "tok_s": 4835.0,
        "needle_rank_64": 3208.0,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_buffer_w8.pt"
    },
    {
        "experiment_id": "exp_buffer_w32",
        "category": "Memory Buffer",
        "initial_ce": 4.7333,
        "final_ce": 3.6964,
        "ce_delta": -1.0369,
        "final_ppl": 40.30,
        "tok_s": 4868.0,
        "needle_rank_64": 3311.2,
        "checkpoint": r"experiments\architecture_matrix\ckpt_exp_buffer_w32.pt"
    },
]

# Sort by final CE
known.sort(key=lambda x: x["final_ce"])

db = {"experiments": {e["experiment_id"]: e for e in known}, "leaderboard": known}
with open(db_path, "w", encoding="utf-8") as f:
    json.dump(db, f, indent=2)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# JARVIS-600M SCIENTIFIC RESEARCH LEADERBOARD\n\n")
    f.write("All experiments evaluated on frozen 606M parameter budget under standardized validation harness.\n\n")
    f.write("| Rank | Experiment ID | Category | Initial CE | Final CE | $\\Delta$ CE | Final PPL | Throughput | Needle Rank @ 64 |\n")
    f.write("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for i, e in enumerate(known, 1):
        f.write(f"| {i} | `{e['experiment_id']}` | {e['category']} | {e['initial_ce']:.4f} | **{e['final_ce']:.4f}** | {e['ce_delta']:+.4f} | {e['final_ppl']:.2f} | {e['tok_s']:.0f} tok/s | {e['needle_rank_64']:.1f} |\n")

print(f"Updated leaderboard with {len(known)} experiments.")
