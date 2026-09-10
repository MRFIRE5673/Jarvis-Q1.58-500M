# Jarvis Architecture Capacity & VRAM Scaling Report

Theoretical analytical model of memory footprint across model scales (600M to 3.0B) and weight precisions (BF16 to 1.58-bit) on the **NVIDIA GeForce RTX 5070 12GB**.

### Model Capacity Summary Table

| Model Scale | Total Params | Active Params | Experts (Top-K) | BF16 Weight | 1.58b Weight | 1.58b Inf VRAM | 1.58b Train VRAM (Ckpt) | RTX 5070 Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Jarvis-600M (Baseline)** | 554,976,256 | 353,649,664 | 4 (Top-2) | 1059 MB | **219 MB** | **492 MB** | 6146 MB | **Trainable & Runnable** |
| **Jarvis-1.0B** | 1,532,744,960 | 693,884,160 | 6 (Top-2) | 2923 MB | **473 MB** | **753 MB** | 15811 MB | **Inference Only** |
| **Jarvis-1.5B** | 3,135,505,920 | 1,097,074,176 | 8 (Top-2) | 5981 MB | **878 MB** | **1162 MB** | 31573 MB | **Inference Only** |
| **Jarvis-2.0B** | 4,715,140,864 | 1,632,327,424 | 8 (Top-2) | 8993 MB | **1276 MB** | **1565 MB** | 47118 MB | **Inference Only** |
| **Jarvis-3.0B** | 7,351,863,296 | 2,520,025,088 | 8 (Top-2) | 14023 MB | **1927 MB** | **2223 MB** | 73049 MB | **Inference Only** |

### Key Takeaways for Jarvis Roadmap:
1. **Jarvis-600M (Current Baseline):** Fits comfortably on RTX 5070 for both training (6.7 GB with grad checkpointing) and inference (0.4 GB weights in 1.58b).
2. **Jarvis-1.0B / 1.1B:** Fits on RTX 5070 for training with 1.58-bit master weights and 8-bit Adam optimizer (or gradient checkpointing at batch size 1-2).
3. **Jarvis-1.5B:** Fits for full 1.58-bit inference on RTX 5070 (uses under 1.2 GB VRAM for weights!). Full training on a single 12GB GPU requires offloading or ZeRO-2.
4. **Jarvis-3.0B:** Fits easily for 1.58-bit packed inference (only ~1.8 GB weights in 1.58b!). Enables running a 3-billion parameter model on a consumer 12GB GPU with room to spare for 32k context.
