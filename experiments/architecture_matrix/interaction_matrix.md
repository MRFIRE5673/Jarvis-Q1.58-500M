# PAIRWISE MEMORY INTERACTION MATRIX

**Baseline CE**: 3.2858

| Pair | Mechanism 1 | Mechanism 2 | Single 1 CE | Single 2 CE | Actual Pair CE | Expected $\Delta$ | Actual $\Delta$ | Interaction | Classification |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `A+B` | Adaptive Decay | Write Gate | 3.2982 | 3.2960 | **3.2983** | +0.0226 | +0.0125 | -0.0101 | **SYNERGISTIC** |
| `A+C` | Adaptive Decay | Erase Gate | 3.2982 | 3.2889 | **3.2895** | +0.0156 | +0.0037 | -0.0119 | **SYNERGISTIC** |
| `A+D` | Adaptive Decay | Gated Read | 3.2982 | 3.2966 | **3.2983** | +0.0233 | +0.0125 | -0.0108 | **SYNERGISTIC** |
| `A+E` | Adaptive Decay | Local Buffer | 3.2982 | 3.2960 | **3.2978** | +0.0226 | +0.0120 | -0.0106 | **SYNERGISTIC** |
| `B+C` | Write Gate | Erase Gate | 3.2960 | 3.2889 | **3.2882** | +0.0134 | +0.0024 | -0.0110 | **SYNERGISTIC** |
| `B+D` | Write Gate | Gated Read | 3.2960 | 3.2966 | **3.2965** | +0.0210 | +0.0107 | -0.0103 | **SYNERGISTIC** |
| `B+E` | Write Gate | Local Buffer | 3.2960 | 3.2960 | **3.2943** | +0.0204 | +0.0085 | -0.0119 | **SYNERGISTIC** |
| `C+D` | Erase Gate | Gated Read | 3.2889 | 3.2966 | **3.2889** | +0.0140 | +0.0031 | -0.0109 | **SYNERGISTIC** |
| `C+E` | Erase Gate | Local Buffer | 3.2889 | 3.2960 | **3.2877** | +0.0133 | +0.0019 | -0.0115 | **SYNERGISTIC** |
| `D+E` | Gated Read | Local Buffer | 3.2966 | 3.2960 | **3.2949** | +0.0210 | +0.0091 | -0.0119 | **SYNERGISTIC** |
