# V-JEPA tri-modal fusion OpenCL profile (L40S)

- device: `Portable Computing Language | NVIDIA L40S | ALL | GPU`
- path: `ProjectedTriModalCrossAttention video/gaze/IMU`
- shapes: Nv=256 Ng=100 Ni=26 D=64 H=4 layers=1
- peak BW: 2953.8 GB/s · peak FLOP: 42253.8 GFLOP/s · ridge AI: 14.305 FLOP/Byte

## Tick totals (steady, weights on device, D2H outs)

| wall_ms | H2D_ms | D2H_ms | IO_ms | launch_ms | device_ms | launches |
|---:|---:|---:|---:|---:|---:|---:|
| 16.128 | 0.000 | 0.013 | 0.013 | 0.872 | 12.136 | 118 |

## Per-kernel (launch = queued→start, device = start→end, T=device for AI/BW/FLOP)

| kernel | # | launch_ms | device_ms | FLOPs | Bytes | AI | BW_eff GB/s | GFLOP/s | bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sdpa_cross_rows` | 3 | 0.022 | 5.059 | 28189624 | 107273216 | 0.263 | 21.20 | 5.57 | latency-bound (low util) |
| `sdpa_rows` | 1 | 0.007 | 4.152 | 26493952 | 100794368 | 0.263 | 24.27 | 6.38 | latency-bound (low util) |
| `topk_indices_row` | 1 | 0.007 | 1.339 | 34816 | 16448 | 2.117 | 0.01 | 0.03 | latency-bound (low util) |
| `softmax_rows` | 16 | 0.119 | 0.596 | 327680 | 1048576 | 0.312 | 1.76 | 0.55 | latency-bound (low util) |
| `qk_logits_chunk` | 16 | 0.120 | 0.489 | 8454144 | 33816576 | 0.250 | 69.13 | 17.28 | latency-bound (low util) |
| `gemm_nn` | 30 | 0.221 | 0.250 | 34724736 | 138898944 | 0.250 | 555.85 | 138.96 | memory-bound |
| `reduce_chunk_to_imp` | 16 | 0.119 | 0.060 | 65536 | 294912 | 0.222 | 4.88 | 1.08 | latency-bound (low util) |
| `d2d_copy_fuse` | 17 | 0.124 | 0.057 | 0 | 1240064 | 0.000 | 21.78 | 0.00 | memory-bound |
| `layernorm_rows` | 1 | 0.007 | 0.037 | 131840 | 393216 | 0.335 | 10.49 | 3.52 | latency-bound (low util) |
| `l2_normalize_rows` | 1 | 0.007 | 0.023 | 49408 | 196608 | 0.251 | 8.47 | 2.13 | latency-bound (low util) |
| `rope_apply` | 1 | 0.007 | 0.022 | 90112 | 200704 | 0.449 | 9.24 | 4.15 | latency-bound (low util) |
| `gelu` | 4 | 0.030 | 0.013 | 408320 | 326656 | 1.250 | 26.08 | 32.60 | latency-bound (low util) |
| `gated_residual` | 3 | 0.022 | 0.010 | 122240 | 391168 | 0.312 | 38.82 | 12.13 | latency-bound (low util) |
| `residual_add` | 3 | 0.022 | 0.009 | 24448 | 293376 | 0.083 | 30.77 | 2.56 | latency-bound (low util) |
| `d2d_copy_Q` | 1 | 0.007 | 0.004 | 0 | 131072 | 0.000 | 35.77 | 0.00 | memory-bound |
| `gather_tokens` | 1 | 0.007 | 0.003 | 0 | 8256 | 0.000 | 2.53 | 0.00 | memory-bound |
| `d2d_zero_imp` | 1 | 0.007 | 0.003 | 0 | 2048 | 0.000 | 0.65 | 0.00 | memory-bound |
| `recency_reweight` | 1 | 0.007 | 0.003 | 768 | 3072 | 0.250 | 0.98 | 0.24 | latency-bound (low util) |
| `fill_recency` | 1 | 0.007 | 0.003 | 512 | 1024 | 0.500 | 0.34 | 0.17 | latency-bound (low util) |

Raw JSON: `scripts/opencl_kernels/profile_system_gpu_N256.json`
Figure: `scripts/opencl_kernels/figures/profile_system_gpu_N256_roofline.png`
