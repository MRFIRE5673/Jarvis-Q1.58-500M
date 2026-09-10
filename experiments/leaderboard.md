# JARVIS-600M RESEARCH LEADERBOARD

| Experiment ID | Category | Initial CE | Final CE | $\Delta$ CE | Final PPL | Throughput | Needle Rank @ 64 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline` | Frozen Baseline | 3.2858 | **3.2858** | +0.0000 | 26.73 | 890 tok/s | 4053.0 |
| `fact_B+C` | Factorial (B+C) | 3.2853 | **3.2882** | +0.0029 | 26.79 | 1213 tok/s | 9085.4 |
| `exp_adaptive_decay` | Adaptive Decay | 3.3412 | **3.2887** | -0.0525 | 26.81 | 5132 tok/s | 10382.2 |
| `fact_C+D` | Factorial (C+D) | 3.2853 | **3.2889** | +0.0036 | 26.81 | 1954 tok/s | 8907.2 |
| `fact_C` | Factorial (C) | 3.2858 | **3.2889** | +0.0032 | 26.81 | 5117 tok/s | 8963.6 |
| `fact_A+C` | Factorial (A+C) | 3.2888 | **3.2895** | +0.0007 | 26.83 | 4903 tok/s | 9200.8 |
| `fact_B+E` | Factorial (B+E) | 3.2836 | **3.2943** | +0.0107 | 26.96 | 2246 tok/s | 9104.8 |
| `fact_E` | Factorial (E) | 3.2960 | **3.2960** | +0.0000 | 27.00 | 3804 tok/s | 9184.4 |
| `fact_B` | Factorial (B) | 3.2847 | **3.2960** | +0.0113 | 27.00 | 4957 tok/s | 8986.2 |
| `fact_B+D` | Factorial (B+D) | 3.2843 | **3.2965** | +0.0123 | 27.02 | 2791 tok/s | 9129.6 |
| `fact_D` | Factorial (D) | 3.2849 | **3.2966** | +0.0117 | 27.02 | 4968 tok/s | 9121.4 |
| `fact_A+E` | Factorial (A+E) | 3.2932 | **3.2978** | +0.0046 | 27.05 | 2422 tok/s | 9193.6 |
| `fact_A` | Factorial (A) | 3.2982 | **3.2982** | +0.0000 | 27.06 | 5169 tok/s | 8927.0 |
| `fact_A+D` | Factorial (A+D) | 3.2947 | **3.2983** | +0.0036 | 27.07 | 1592 tok/s | 9102.0 |
| `fact_A+B` | Factorial (A+B) | 3.2947 | **3.2983** | +0.0036 | 27.07 | 1214 tok/s | 9172.0 |
| `exp_aux_free_bias` | MoE Routing | 3.2857 | **3.2985** | +0.0128 | 27.07 | 604 tok/s | 3042.0 |
| `exp_gated_read` | Gated Read | 3.3186 | **3.3001** | -0.0184 | 27.12 | 5133 tok/s | 7482.6 |
| `exp_squared_relu` | FFN Activations | 3.4495 | **3.3049** | -0.1446 | 27.24 | 568 tok/s | 3108.0 |
| `exp_erase_gate` | Erase Gate Memory | 3.5124 | **3.3169** | -0.1956 | 27.57 | 5082 tok/s | 8450.0 |
| `exp_delta_memory` | Associative Memory | 9.4328 | **3.3231** | -6.1097 | 27.75 | 392 tok/s | 7774.6 |
| `exp_write_erase_gate` | Write/Erase Memory | 3.6004 | **3.3232** | -0.2772 | 27.75 | 5047 tok/s | 7821.4 |
| `exp_buffer_write_erase` | Hybrid Buffer Write/Erase | 4.2305 | **3.3286** | -0.9020 | 27.90 | 4591 tok/s | 8591.4 |
| `v2_write_erase` | Combined V2 Write/Erase | 4.9032 | **3.3511** | -1.5520 | 28.54 | 4718 tok/s | 8383.4 |
| `exp_best_memory_combo` | Best Memory Combo | 5.2801 | **3.3689** | -1.9111 | 29.05 | 4545 tok/s | 11671.8 |
| `exp_swiglu` | FFN Activations | 4.5371 | **3.3936** | -1.1435 | 29.78 | 257 tok/s | 3180.0 |
| `exp_deepseek_shared` | MoE Routing | 4.5371 | **3.3992** | -1.1379 | 29.94 | 546 tok/s | 3150.0 |
| `exp_muon_hybrid` | Optimizer Dynamics | 3.2858 | **3.4434** | +0.1576 | 31.29 | 540 tok/s | 3200.0 |
| `v2_ablate_gelu` | V2 Ablation | 5.1134 | **3.5505** | -1.5630 | 34.83 | 4890 tok/s | 4748.8 |
| `v2_full` | Combined V2 | 5.2262 | **3.5601** | -1.6661 | 35.17 | 4805 tok/s | 4942.8 |
| `exp_sliding_buffer` | Attention & Recurrence | 4.7329 | **3.6792** | -1.0537 | 39.61 | 863 tok/s | 2016.0 |
| `v2_ablate_nobuffer` | V2 Ablation | 4.7708 | **3.6855** | -1.0853 | 39.86 | 5086 tok/s | 2930.2 |
| `exp_buffer_w8` | Memory Buffer | 4.7336 | **3.6951** | -1.0385 | 40.25 | 4835 tok/s | 3208.0 |
| `exp_buffer_w32` | Memory Buffer | 4.7333 | **3.6964** | -1.0369 | 40.30 | 4868 tok/s | 3311.2 |
| `exp_gated_write` | Attention & Recurrence | 5.2263 | **3.7341** | -1.4922 | 41.85 | 950 tok/s | 2664.0 |
| `v2_ablate_nogate` | V2 Ablation | 5.2084 | **3.7857** | -1.4227 | 44.07 | 4869 tok/s | 2677.4 |
| `exp_multi_timescale` | Attention & Recurrence | 5.4192 | **3.8256** | -1.5935 | 45.86 | 870 tok/s | 1831.0 |
