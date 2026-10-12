"""
Microarchitectural Hardware Simulator & Numerical Engine
Models SRAM vs. DRAM (HBM) data movement in Transformer Attention.
"""

import math
import numpy as np


class AttentionHardwareModel:
    def __init__(self, sram_kb: float = 192.0, dram_bw_gbs: float = 2039.0, peak_tflops: float = 312.0):
        self.sram_bytes = sram_kb * 1024.0
        self.dram_bw_bytes_sec = dram_bw_gbs * 1e9
        self.peak_flops_sec = peak_tflops * 1e12

    def calculate_metrics(self, seq_len: int, head_dim: int, batch_heads: int, precision_bytes: int = 2):
        # 1. Total Arithmetic Floating-Point Operations
        # Q*K^T is 2 * N^2 * d, Softmax * V is 2 * N^2 * d
        flops_per_head = 4 * (seq_len ** 2) * head_dim
        total_flops = flops_per_head * batch_heads

        # 2. Naive Unfused Attention DRAM Traffic (Bytes)
        # Reads Q, K, V: 3 * N * d
        # Writes S (N x N) to DRAM: N * N
        # Reads S, Writes P (Softmax): 2 * N * N
        # Reads P, Reads V, Writes O: N * N + N * d + N * d
        unfused_elements_per_head = (4 * seq_len * head_dim) + (3 * (seq_len ** 2))
        unfused_bytes = unfused_elements_per_head * precision_bytes * batch_heads

        # 3. Fused / Flash-Style Attention DRAM Traffic (Bytes)
        # Optimal block size B fitting Q_block, K_block, V_block, O_block in SRAM
        tile_size = int(self.sram_bytes / (4 * head_dim * precision_bytes))
        tile_size = max(16, min(tile_size, 256))

        # I/O Complexity: Avoids materializing N x N intermediate matrices in DRAM
        # DRAM Traffic = 2 * N * d (Read Q, Write O) + (2 * N^2 * d * precision) / SRAM
        fused_bytes_per_head = (
            2 * seq_len * head_dim * precision_bytes
            + (2 * (seq_len ** 2) * head_dim * (precision_bytes ** 2)) / self.sram_bytes
        )
        # Bound fused bytes by naive unfused in small sequences
        fused_bytes_per_head = min(fused_bytes_per_head, unfused_elements_per_head * precision_bytes)
        fused_bytes = fused_bytes_per_head * batch_heads

        # 4. Latency Estimation (Memory vs Compute Bound)
        compute_time_sec = total_flops / self.peak_flops_sec
        unfused_mem_time_sec = unfused_bytes / self.dram_bw_bytes_sec
        fused_mem_time_sec = fused_bytes / self.dram_bw_bytes_sec

        # Execution time dominated by bottleneck
        unfused_latency_ms = max(compute_time_sec, unfused_mem_time_sec) * 1000
        fused_latency_ms = max(compute_time_sec, fused_mem_time_sec) * 1000
        speedup = unfused_latency_ms / max(fused_latency_ms, 1e-9)
        io_reduction = unfused_bytes / max(fused_bytes, 1e-9)

        intermediate_s_mb = (seq_len ** 2 * precision_bytes) / 1e6

        return {
            "total_flops": total_flops,
            "intermediate_s_mb": intermediate_s_mb,
            "unfused_bytes_gb": unfused_bytes / 1e9,
            "fused_bytes_gb": fused_bytes / 1e9,
            "unfused_latency_ms": unfused_latency_ms,
            "fused_latency_ms": fused_latency_ms,
            "speedup": speedup,
            "io_reduction": io_reduction,
            "tile_size": tile_size,
        }

    @staticmethod
    def verify_online_softmax():
        """Numerically validates that block-tiled online softmax matches global standard softmax."""
        np.random.seed(42)
        dim = 16
        scores = np.random.randn(dim)

        # Standard 2-Pass Softmax (Baseline)
        exp_standard = np.exp(scores - np.max(scores))
        standard_out = exp_standard / np.sum(exp_standard)

        # Flash-Style 1-Pass Online Softmax (in tiles of 4)
        tile_size = 4
        m_prev = -float("inf")
        d_prev = 0.0
        acc = np.zeros(dim)

        num_tiles = dim // tile_size
        for i in range(num_tiles):
            chunk = scores[i * tile_size : (i + 1) * tile_size]
            m_curr = max(m_prev, float(np.max(chunk)))
            
            # Rescale running sum of exponents
            alpha = np.exp(m_prev - m_curr) if m_prev != -float("inf") else 0.0
            d_prev = d_prev * alpha + np.sum(np.exp(chunk - m_curr))
            
            # Store unnormalized scaled exponentials
            acc[i * tile_size : (i + 1) * tile_size] = np.exp(chunk - m_curr)
            m_prev = m_curr

        # Final normalization pass across cached registers
        online_out = acc / d_prev
        max_abs_diff = float(np.max(np.abs(standard_out - online_out)))
        is_valid = max_abs_diff < 1e-6

        return is_valid, max_abs_diff, standard_out, online_out