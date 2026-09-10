# Jarvis Research Sprint: Unified Architecture Leaderboard

**Git Commit:** `6a30b2741446159efb31755e77af72e3d6f31858`  
**Locked Baseline Checkpoint:** `experiments/extended_train/ckpt_step_0004284_best.pt`  
**Packed 1.58b Checkpoint:** `experiments/architecture_matrix/ternary_packed/ckpt_baseline_packed_158b.pt`  

| Architecture | Parameters | Overhead | State (KB/seq) | File Size | CE@512 | PPL@512 | CE@8192 | Needle Rank | Speed (tok/s) | Peak VRAM | Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Paper Baseline** | 606,391,704 | +0 | 6144 KB | 2314.2 MB | 2.8511 | 17.31 | 3.2721 | 10476.8 | 2209.0 | 2691 MB | **BASELINE REFERENCE** |
| **E-W8 (Window=8)** | 606,392,088 | +384 | 6144 KB | 2314.2 MB | 3.0447 | 21.00 | 3.2708 | 7121.1 | 4652.9 | 10293 MB | **PROMISING** |
| **E-W16 (Window=16)** | 606,392,088 | +384 | 6144 KB | 2314.2 MB | 3.0447 | 21.00 | 3.2707 | 7222.3 | 4664.7 | 10293 MB | **KEEP** |
| **E-W32 (Window=32)** | 606,392,088 | +384 | 6144 KB | 2314.2 MB | 3.0449 | 21.01 | 3.2706 | 7247.1 | 4982.7 | 10293 MB | **PROMISING** |
| **Multi-Scale (W={8,16,32})** | 606,392,088 | +384 | 6144 KB | 2314.2 MB | 3.0448 | 21.01 | 3.2707 | 7102.5 | 5041.9 | 10293 MB | **KEEP (BEST SPEED & RETRIEVAL)** |
| **State-Compacted (GRM)** | 568,643,064 | -37,748,640 | 1536 KB | 2169.5 MB | 3.2958 | 27.00 | 5.8239 | 5726.0 | 4762.1 | 9713 MB | **PROMISING (NEEDS PRETRAIN)** |
| **Packed-Ternary 1.58b (Storage + Kernel)** | 606,391,704 | +0 | 6144 KB | 317.3 MB | 2.8511 | 17.31 | 3.2721 | 10476.8 | 12244.1 | 2690 MB | **KEEP (FOUNDATION LOCKED)** |
