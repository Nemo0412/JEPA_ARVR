// V-JEPA encoder / predictor block primitives (OpenCL baselines)
// Pipeline: LN -> QKV -> axial RoPE -> SDPA -> out_proj -> +res -> LN -> MLP -> +res
// Shapes use flat layouts noted per kernel. Research prototypes, not FlashAttention.

#ifndef GELU_SCALE
#define GELU_SCALE 0.7978845608028654f /* sqrt(2/pi) */
#endif

__kernel void layernorm_rows(
    __global const float *x,   // [rows, D]
    __global float *y,
    __global const float *gamma, // [D] or NULL-equivalent: pass ones
    __global const float *beta,  // [D]
    const int rows,
    const int D,
    const float eps
) {
    const int r = get_global_id(0);
    if (r >= rows) return;
    __global const float *src = x + r * D;
    __global float *dst = y + r * D;
    float mean = 0.0f;
    for (int d = 0; d < D; ++d) mean += src[d];
    mean /= (float)D;
    float var = 0.0f;
    for (int d = 0; d < D; ++d) {
        float v = src[d] - mean;
        var += v * v;
    }
    var /= (float)D;
    float inv = rsqrt(var + eps);
    for (int d = 0; d < D; ++d) {
        float n = (src[d] - mean) * inv;
        dst[d] = n * gamma[d] + beta[d];
    }
}

// y[m,n] = sum_k A[m,k] * B[k,n]   A:[M,K] B:[K,N]  (row-major)
__kernel void gemm_nn(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta
) {
    const int gid = get_global_id(0);
    const int total = M * N;
    if (gid >= total) return;
    const int n = gid % N;
    const int m = gid / N;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
    }
    C[gid] = alpha * acc + beta * C[gid];
}

// Apply V-JEPA-style pair rotate+RoPE on last dim of Q or K.
// x: [rows, Dh] in/out (rows=B*H*N); pos: [rows].
// Pretrained-compatible duplicated freqs (torch .repeat, not repeat_interleave).
__kernel void rope_apply(
    __global float *x,
    __global const float *pos,
    const int rows,
    const int Dh
) {
    const int r = get_global_id(0);
    if (r >= rows) return;
    if ((Dh & 1) != 0) return;
    const int dh2 = Dh / 2;
    if (dh2 > 128) return;
    const float p = pos[r];
    __global float *v = x + r * Dh;

    float cos_h[128];
    float sin_h[128];
    for (int i = 0; i < dh2; ++i) {
        float omega = pow(10000.0f, -((float)i / (float)dh2));
        float freq = p * omega;
        cos_h[i] = cos(freq);
        sin_h[i] = sin(freq);
    }
    // duplicate whole half-vector: [c0..c_{h-1}, c0..c_{h-1}]
    float cos_full[256];
    float sin_full[256];
    for (int i = 0; i < dh2; ++i) {
        cos_full[i] = cos_h[i];
        sin_full[i] = sin_h[i];
        cos_full[dh2 + i] = cos_h[i];
        sin_full[dh2 + i] = sin_h[i];
    }

    float y[256];
    for (int i = 0; i < dh2; ++i) {
        float x0 = v[2 * i];
        float x1 = v[2 * i + 1];
        y[2 * i] = -x1;
        y[2 * i + 1] = x0;
    }
    for (int d = 0; d < Dh; ++d) {
        v[d] = v[d] * cos_full[d] + y[d] * sin_full[d];
    }
}

// Naive SDPA: one WI per (b,h,i) query row.
// Q,K,V: [B,H,N,Dh]; Out: [B,H,N,Dh]
__kernel void sdpa_rows(
    __global const float *Q,
    __global const float *K,
    __global const float *V,
    __global float *O,
    const int B,
    const int H,
    const int N,
    const int Dh,
    const float scale
) {
    const int gid = get_global_id(0);
    const int rows = B * H * N;
    if (gid >= rows) return;
    const int i = gid % N;
    const int tmp = gid / N;
    const int h = tmp % H;
    const int b = tmp / H;

    const int head_stride = N * Dh;
    const int batch_stride = H * head_stride;
    __global const float *q = Q + b * batch_stride + h * head_stride + i * Dh;
    __global const float *kbase = K + b * batch_stride + h * head_stride;
    __global const float *vbase = V + b * batch_stride + h * head_stride;
    __global float *o = O + b * batch_stride + h * head_stride + i * Dh;

    float m = -INFINITY;
    for (int j = 0; j < N; ++j) {
        __global const float *kj = kbase + j * Dh;
        float dot = 0.0f;
        for (int d = 0; d < Dh; ++d) dot += q[d] * kj[d];
        m = fmax(m, dot * scale);
    }
    float z = 0.0f;
    for (int d = 0; d < Dh; ++d) o[d] = 0.0f;
    for (int j = 0; j < N; ++j) {
        __global const float *kj = kbase + j * Dh;
        __global const float *vj = vbase + j * Dh;
        float dot = 0.0f;
        for (int d = 0; d < Dh; ++d) dot += q[d] * kj[d];
        float a = exp(dot * scale - m);
        z += a;
        for (int d = 0; d < Dh; ++d) o[d] += a * vj[d];
    }
    float inv = 1.0f / z;
    for (int d = 0; d < Dh; ++d) o[d] *= inv;
}

__kernel void residual_add(
    __global const float *a,
    __global const float *b,
    __global float *out,
    const int n
) {
    const int i = get_global_id(0);
    if (i < n) out[i] = a[i] + b[i];
}

// GELU (tanh approx) elementwise
__kernel void gelu(
    __global const float *x,
    __global float *y,
    const int n
) {
    const int i = get_global_id(0);
    if (i >= n) return;
    float v = x[i];
    float u = GELU_SCALE * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.0f + tanh(u));
}

// Cross-attention SDPA: Q [B,H,Nq,Dh], K/V [B,H,Nk,Dh]
__kernel void sdpa_cross_rows(
    __global const float *Q,
    __global const float *K,
    __global const float *V,
    __global float *O,
    const int B,
    const int H,
    const int Nq,
    const int Nk,
    const int Dh,
    const float scale
) {
    const int gid = get_global_id(0);
    const int rows = B * H * Nq;
    if (gid >= rows) return;
    const int i = gid % Nq;
    const int tmp = gid / Nq;
    const int h = tmp % H;
    const int b = tmp / H;

    const int q_head = Nq * Dh;
    const int k_head = Nk * Dh;
    __global const float *q = Q + ((b * H + h) * Nq + i) * Dh;
    __global const float *kbase = K + (b * H + h) * Nk * Dh;
    __global const float *vbase = V + (b * H + h) * Nk * Dh;
    __global float *o = O + ((b * H + h) * Nq + i) * Dh;

    float m = -INFINITY;
    for (int j = 0; j < Nk; ++j) {
        __global const float *kj = kbase + j * Dh;
        float dot = 0.0f;
        for (int d = 0; d < Dh; ++d) dot += q[d] * kj[d];
        m = fmax(m, dot * scale);
    }
    float z = 0.0f;
    for (int d = 0; d < Dh; ++d) o[d] = 0.0f;
    for (int j = 0; j < Nk; ++j) {
        __global const float *kj = kbase + j * Dh;
        __global const float *vj = vbase + j * Dh;
        float dot = 0.0f;
        for (int d = 0; d < Dh; ++d) dot += q[d] * kj[d];
        float a = exp(dot * scale - m);
        z += a;
        for (int d = 0; d < Dh; ++d) o[d] += a * vj[d];
    }
    float inv = 1.0f / z;
    for (int d = 0; d < Dh; ++d) o[d] *= inv;
}

// Gate residual: out = z + sigmoid(gate(z, attn)) * (attn - z) style used in fusion
// Here: out = z + sigmoid(g) * delta, with g,z,delta all [n]
__kernel void gated_residual(
    __global const float *z,
    __global const float *delta,
    __global const float *gate_logit,
    __global float *out,
    const int n
) {
    const int i = get_global_id(0);
    if (i >= n) return;
    float g = 1.0f / (1.0f + exp(-gate_logit[i]));
    out[i] = z[i] + g * delta[i];
}
