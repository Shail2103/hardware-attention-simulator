# Hardware I/O-Aware Attention & Memory-Traffic Simulator

A microarchitectural performance modeling tool designed to analyze **DRAM (HBM) bandwidth bottlenecks**, **on-chip SRAM constraints**, and **kernel fusion speedups** in Transformer self-attention workloads.

Built with Python and Streamlit, this tool simulates the transition from standard memory-bound attention to compute-bound tiled attention (Flash-style online softmax) across industry GPU architectures (NVIDIA H100 Hopper, A100 Ampere, RTX 4090, Jetson Orin) and arbitrary custom silicon profiles[cite: 1].

---

##  Live Interactive Dashboard
Experience the live microarchitectural simulator directly in your browser:  
**[Launch Interactive Hardware Simulator](https://attention-io-simulator.streamlit.app/)** 

---

## Key Highlights

- **Microarchitectural Modeling:** Analyzes operational intensity ($\text{FLOPs/Byte}$), off-chip DRAM memory transfers, and SRAM capacity limits ($32\text{ KB} - 512\text{ KB}$) across configurable batch sizes, head dimensions, and sequence lengths ($N = 256$ to $8192$)[cite: 3].
- **Memory Wall Demonstration:** Illustrates the quadratic scaling bottleneck ($\mathcal{O}(N^2)$ DRAM traffic) caused by materializing intermediate attention matrices ($\mathbf{S}$ and $\mathbf{P}$) in global memory[cite: 3].
- **SRAM Kernel Fusion:** Simulates block-level tiling and I/O reduction ($\approx 8\times$ memory traffic savings) enabled by fused online softmax execution[cite: 3].
- **Numerical Invariance Engine:** Includes a micro-verification unit demonstrating that incremental online softmax maintains numerical equivalence ($\Delta < 10^{-7}$) without computing full global matrix row denominators upfront.
- **Dynamic Hardware Presets:** Toggle between NVIDIA production silicon targets (Hopper H100, Ampere A100) or specify custom experimental accelerator parameters (SRAM/SM, memory bandwidth, peak compute).

---

## Architectural Background: The Attention Memory Wall

In conventional deep learning frameworks, scaled dot-product attention executes via three separate kernel invocations[cite: 3]:

$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T \quad \rightarrow \quad \mathbf{P} = \text{softmax}(\mathbf{S}) \quad \rightarrow \quad \mathbf{O} = \mathbf{P}\mathbf{V}$$

1. **Standard (Unfused) Attention:**
   - Writes the full $N \times N$ matrix $\mathbf{S}$ out to off-chip DRAM (HBM)[cite: 3].
   - Reads $\mathbf{S}$ back into SRAM to compute softmax, and writes intermediate probabilities $\mathbf{P}$ back to DRAM[cite: 3].
   - Reads $\mathbf{P}$ and $\mathbf{V}$ from DRAM to compute final projection $\mathbf{O}$[cite: 3].
   - **Total DRAM I/O Complexity:** $\mathcal{O}(N^2)$ — leads to compute starvation on Tensor Cores[cite: 3].

2. **Fused Tiled Attention (Flash-Style):**
   - Loads small blocks $B_r \times d$ and $B_c \times d$ directly into fast on-chip SRAM[cite: 3].
   - Computes local attention scores, updates running softmax scaling factors dynamically via the **Online Softmax Algorithm**, and updates output accumulators directly in registers/SRAM[cite: 3].
   - Never writes intermediate $N \times N$ activation matrices to DRAM[cite: 3].
   - **Total DRAM I/O Complexity:** $\mathcal{O}(N)$ — shifts execution from memory-bound to compute-bound[cite: 3].

---

## Performance Comparison: Unfused vs. Fused Tiling

| Metric | Standard Unfused Attention | Fused Tiled Attention (Online Softmax) | Architectural Impact |
| :--- | :--- | :--- | :--- |
| **Intermediate $S$ Footprint** | Materialized in DRAM (${O}(N^2)$) | Never materialized off-chip | Zero DRAM allocation for logits |
| **DRAM Memory Traffic** | $(4Nd + 3N^2) \times \text{bytes}$ | $\approx 2Nd + \frac{2N^2 d \times \text{bytes}}{S_{\text{SRAM}}}$ | Up to **$8\times$ reduction** in DRAM I/O[cite: 3] |
| **Execution Regime** | **Memory-Bound** (DRAM Starved) | **Compute-Bound** (Tensor Core Saturated) | Minimizes memory stall cycles[cite: 3] |
| **Numerical Accuracy** | Two-pass standard softmax | Dynamically rescaled single-pass | Exact floating-point parity |

---

## Repository Structure

```text
├── attention_engine.py    # Analytical microarchitectural model & numerical invariance engine
├── dashboard.py           # Streamlit interactive visualization frontend
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
