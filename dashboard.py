"""
Streamlit Web Frontend with Dynamic Chip Definition Support
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from attention_engine import AttentionHardwareModel

st.set_page_config(
    page_title="Hardware I/O-Aware Attention Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Hardware I/O-Aware Attention Simulator")
st.markdown("Simulating **DRAM Memory Walls** and **On-Chip SRAM Kernel Fusion** across any custom or off-the-shelf architecture.")

# --- SIDEBAR: DYNAMIC HARDWARE TARGET SELECTION ---
st.sidebar.header("1. Target Hardware Architecture")

hardware_mode = st.sidebar.radio(
    "Mode",
    ["Select Pre-existing Preset", "Define Custom / Fictional Chip"],
    index=0
)

GPU_PRESETS = {
    "NVIDIA H100 SXM5 (Hopper)": {
        "sram_kb": 228.0,
        "dram_bw": 3350.0,
        "peak_flops": 989.0,
        "desc": "Hopper architecture with 80GB HBM3 and 4th Gen Tensor Cores"
    },
    "NVIDIA A100 SXM4 (Ampere)": {
        "sram_kb": 192.0,
        "dram_bw": 2039.0,
        "peak_flops": 312.0,
        "desc": "Ampere architecture with 80GB HBM2e and 3rd Gen Tensor Cores"
    },
    "NVIDIA RTX 4090 (Ada Lovelace)": {
        "sram_kb": 128.0,
        "dram_bw": 1008.0,
        "peak_flops": 330.0,
        "desc": "Consumer flagship with GDDR6X memory and 4th Gen Tensor Cores"
    },
    "NVIDIA Jetson AGX Orin (Edge)": {
        "sram_kb": 64.0,
        "dram_bw": 204.0,
        "peak_flops": 170.0,
        "desc": "Low-power embedded SoC for edge robotics and autonomous systems"
    }
}

if hardware_mode == "Select Pre-existing Preset":
    preset_name = st.sidebar.selectbox("Select GPU Preset", list(GPU_PRESETS.keys()))
    spec = GPU_PRESETS[preset_name]
    chip_name = preset_name
    chip_desc = spec["desc"]
    sram_kb = spec["sram_kb"]
    dram_bw = spec["dram_bw"]
    peak_flops = spec["peak_flops"]
    
    st.sidebar.caption(f"ℹ️ {chip_desc}")
    st.sidebar.metric("SRAM / Shared Memory", f"{sram_kb:.0f} KB / SM")
    st.sidebar.metric("Memory Bandwidth", f"{dram_bw:,.0f} GB/s")
    st.sidebar.metric("Peak FP16 Compute", f"{peak_flops:.0f} TFLOPs")

else:
    # Custom Chip Definition
    st.sidebar.markdown("### 🛠️ Custom Chip Specifications")
    chip_name = st.sidebar.text_input("Chip Name", value="NeuroCore-X1 Prototype")
    chip_desc = st.sidebar.text_area("Description / Use Case", value="Custom experimental neural processing accelerator with high on-chip cache.")
    
    sram_kb = st.sidebar.number_input("On-Chip SRAM per Compute Core (KB)", min_value=16.0, max_value=2048.0, value=256.0, step=16.0)
    dram_bw = st.sidebar.number_input("Main Memory (DRAM/HBM) Bandwidth (GB/s)", min_value=10.0, max_value=8000.0, value=1200.0, step=50.0)
    peak_flops = st.sidebar.number_input("Peak Matrix Compute (TFLOPs)", min_value=1.0, max_value=3000.0, value=400.0, step=25.0)

# Instantiate the hardware engine using the configured numbers
model = AttentionHardwareModel(sram_kb=sram_kb, dram_bw_gbs=dram_bw, peak_tflops=peak_flops)

# Display active target banner
st.info(f"**Target System:** `{chip_name}` | **SRAM:** `{sram_kb:.0f} KB/SM` | **DRAM Bandwidth:** `{dram_bw:,.0f} GB/s` | **Compute:** `{peak_flops:.0f} TFLOPs`\n\n*{chip_desc}*")

# --- WORKLOAD PARAMETERS ---
st.subheader("2. Attention Workload Configuration")
c1, c2, c3, c4 = st.columns(4)

with c1:
    seq_len = st.select_slider("Sequence Length (N)", options=[256, 512, 1024, 2048, 4096, 8192], value=2048)
with c2:
    head_dim = st.selectbox("Head Dimension (d)", [64, 128], index=0)
with c3:
    prec = st.selectbox("Precision", [2, 4], format_func=lambda x: "FP16 (2 Bytes)" if x == 2 else "FP32 (4 Bytes)")
with c4:
    batch_heads = st.number_input("Batch × Heads", min_value=1, max_value=64, value=16, step=4)

# Calculate hardware metrics
metrics = model.calculate_metrics(seq_len=seq_len, head_dim=head_dim, batch_heads=batch_heads, precision_bytes=prec)

# --- METRIC CARDS ---
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Intermediate S Matrix Footprint", f"{metrics['intermediate_s_mb']:.1f} MB/head")
m2.metric("Unfused DRAM Access", f"{metrics['unfused_bytes_gb']:.2f} GB")
m3.metric("Fused Tiled DRAM Access", f"{metrics['fused_bytes_gb']:.2f} GB")
m4.metric("DRAM Traffic Reduction", f"{metrics['io_reduction']:.1f}x", delta=f"{metrics['speedup']:.2f}x Speedup")

# --- ARCHITECTURAL CHARTS ---
st.subheader(f"3. Performance & Memory Wall Scaling on `{chip_name}`")
p1, p2 = st.columns(2)

seq_range = [256, 512, 1024, 2048, 4096, 8192]
unfused_curve = []
fused_curve = []

for s in seq_range:
    res = model.calculate_metrics(seq_len=s, head_dim=head_dim, batch_heads=batch_heads, precision_bytes=prec)
    unfused_curve.append(res["unfused_bytes_gb"])
    fused_curve.append(res["fused_bytes_gb"])

with p1:
    fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=120)
    ax1.plot(seq_range, unfused_curve, "r-o", label="Unfused (DRAM Materialization)", linewidth=2)
    ax1.plot(seq_range, fused_curve, "g-s", label="Fused / Flash (SRAM Tiled)", linewidth=2)
    ax1.set_xlabel("Sequence Length (N)", fontsize=10)
    ax1.set_ylabel("Total DRAM Traffic (GB)", fontsize=10)
    ax1.set_title(f"Memory Wall Scaling on {chip_name}", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    st.pyplot(fig1)

with p2:
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=120)
    labels = ["Unfused Attention", "Fused Attention"]
    latencies = [metrics["unfused_latency_ms"], metrics["fused_latency_ms"]]
    colors = ["#d9534f", "#5cb85c"]
    bars = ax2.bar(labels, latencies, color=colors, width=0.45)
    ax2.set_ylabel("Simulated Execution Latency (ms)", fontsize=10)
    ax2.set_title(f"Execution Latency at N = {seq_len}", fontsize=11)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.2f} ms", ha="center", va="bottom", fontweight="bold")
    st.pyplot(fig2)

# --- VERIFICATION TEST ---
st.markdown("---")
st.subheader("4. Online Softmax Invariance Engine")
st.caption("Verifying that block-tiled online softmax remains numerically exact regardless of the hardware configuration.")

if st.button("Run Numerical Equivalence Test"):
    passed, error, std_res, onl_res = AttentionHardwareModel.verify_online_softmax()
    if passed:
        st.success(f"Verification Passed! Max Absolute Error: {error:.2e} (Strictly within floating-point epsilon).")
    else:
        st.error(f"Numerical Discrepancy Detected: {error:.2e}")