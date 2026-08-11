// Idea1: fused receive-importance (no logits/attn materialization)
// Idea2 helpers: cheap CA-side token scores

// Portable float atomic add (OpenCL 1.2)
inline void atomic_add_float(__global float *addr, float val) {
    __global volatile uint *uaddr = (__global volatile uint *)addr;
    union { uint u; float f; } oldv, newv;
    uint prev = *uaddr;
    do {
        oldv.u = prev;
        newv.f = oldv.f + val;
        prev = atomic_cmpxchg(uaddr, oldv.u, newv.u);
    } while (prev != oldv.u);
}

// Fused receive (v1): one WI per query — cache dots in private tiles to avoid 3 full global passes.
// TILE keys; for each query recompute only one pass of dots into a register tile then accumulate.
#ifndef FUSED_TILE
#define FUSED_TILE 128
#endif

__kernel void receive_importance_fused(
    __global const float *x,   // [B,N,D] L2-normalized
    __global float *imp,       // [B,N] must be zeroed
    const int B,
    const int N,
    const int D,
    const float scale
) {
    const int gid = get_global_id(0);
    const int total_q = B * N;
    if (gid >= total_q) return;

    const int b = gid / N;
    const int i = gid - b * N;
    const __global float *q = x + ((size_t)b * N + i) * D;
    const __global float *xb = x + (size_t)b * N * D;
    __global float *imp_b = imp + b * N;

    // Pass A: global max (still one full pass — necessary for stability)
    float m = -INFINITY;
    for (int j = 0; j < N; ++j) {
        const __global float *k = xb + (size_t)j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q[d] * k[d];
        m = fmax(m, dot * scale);
    }

    // Pass B: Z + receive accumulate using a small score tile cache (2 passes worth of math,
    // but writes imp without a global [N,N] or [C,N] buffer)
    float z = 0.0f;
    float tile[FUSED_TILE];
    for (int j0 = 0; j0 < N; j0 += FUSED_TILE) {
        int ntile = N - j0;
        if (ntile > FUSED_TILE) ntile = FUSED_TILE;
        for (int t = 0; t < ntile; ++t) {
            const __global float *k = xb + (size_t)(j0 + t) * D;
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += q[d] * k[d];
            float e = exp(dot * scale - m);
            tile[t] = e;
            z += e;
        }
        // cannot normalize until z known — so only accumulate e for Z here; second loop below.
        // Store raw exp in tile only for this tile segment after Z is done requires another key pass.
        (void)tile;
    }
    // full Z then scatter (one more key pass) — same 3-pass, but no [C,N] DRAM traffic
    float inv_z = 0.0f;
    // recompute Z cleanly
    z = 0.0f;
    for (int j = 0; j < N; ++j) {
        const __global float *k = xb + (size_t)j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q[d] * k[d];
        z += exp(dot * scale - m);
    }
    inv_z = 1.0f / z;
    for (int j = 0; j < N; ++j) {
        const __global float *k = xb + (size_t)j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q[d] * k[d];
        float a = exp(dot * scale - m) * inv_z;
        atomic_add_float(&imp_b[j], a);
    }
}

// Idea1 optimized: keep ONE chunk buffer of logits, fuse softmax→imp (no attn buffer, no atomics).
// Work-item = key index j over one query chunk rows via serial loop on C (stable on CPU).
__kernel void softmax_rows_reduce_to_imp(
    __global const float *logits,  // [B, C, N]
    __global float *imp,           // [B, N]
    const int B,
    const int N,
    const int C
) {
    const int gid = get_global_id(0);  // b*N + j
    const int total = B * N;
    if (gid >= total) return;
    const int b = gid / N;
    const int j = gid - b * N;

    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        __global const float *row = logits + ((size_t)(b * C + c) * (size_t)N);
        float m = -INFINITY;
        for (int t = 0; t < N; ++t) m = fmax(m, row[t]);
        float z = 0.0f;
        for (int t = 0; t < N; ++t) z += exp(row[t] - m);
        acc += exp(row[j] - m) / z;
    }
    imp[gid] += acc;
}

// In-place row softmax: logits[row,:] -> softmax in the same buffer
__kernel void softmax_rows_inplace(
    __global float *logits,
    const int rows,
    const int N
) {
    const int row = get_global_id(0);
    if (row >= rows) return;
    __global float *src = logits + (size_t)row * (size_t)N;
    float m = -INFINITY;
    for (int j = 0; j < N; ++j) m = fmax(m, src[j]);
    float z = 0.0f;
    for (int j = 0; j < N; ++j) {
        float e = exp(src[j] - m);
        src[j] = e;
        z += e;
    }
    float inv_z = 1.0f / z;
    for (int j = 0; j < N; ++j) src[j] *= inv_z;
}

// Query-strided fused receive (Idea1+2 light self): only queries i=q0,q0+stride,...
__kernel void receive_importance_fused_strided(
    __global const float *x,
    __global float *imp,
    const int B,
    const int N,
    const int D,
    const float scale,
    const int q_stride,
    const int q0
) {
    const int gid = get_global_id(0);
    // number of active queries per batch
    const int nq = (N - q0 + q_stride - 1) / q_stride;
    if (nq <= 0) return;
    const int total = B * nq;
    if (gid >= total) return;

    const int t = gid % nq;
    const int b = gid / nq;
    const int i = q0 + t * q_stride;
    if (i >= N) return;

    const __global float *q = x + ((size_t)b * N + i) * D;
    const __global float *xb = x + (size_t)b * N * D;
    __global float *imp_b = imp + b * N;

    float m = -INFINITY;
    for (int j = 0; j < N; ++j) {
        const __global float *k = xb + (size_t)j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q[d] * k[d];
        m = fmax(m, dot * scale);
    }
    float z = 0.0f;
    for (int j = 0; j < N; ++j) {
        const __global float *k = xb + (size_t)j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q[d] * k[d];
        z += exp(dot * scale - m);
    }
    float inv_z = 1.0f / z;
    // scale by stride so total mass ~ N instead of N/stride
    float mass = (float)q_stride;
    for (int j = 0; j < N; ++j) {
        const __global float *k = xb + (size_t)j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q[d] * k[d];
        float a = exp(dot * scale - m) * inv_z * mass;
        atomic_add_float(&imp_b[j], a);
    }
}

// Idea2: CA output token score = L2 norm of O (video queries), layout [B,H,Nq,Dh] -> [B,Nq]
__kernel void ca_output_l2_score(
    __global const float *O,  // [B,H,Nq,Dh]
    __global float *score,    // [B,Nq] zeroed then summed over heads (energy)
    const int B,
    const int H,
    const int Nq,
    const int Dh
) {
    const int gid = get_global_id(0);
    const int total = B * H * Nq;
    if (gid >= total) return;
    const int i = gid % Nq;
    const int tmp = gid / Nq;
    const int h = tmp % H;
    const int b = tmp / H;
    const __global float *o = O + (((size_t)b * H + h) * Nq + i) * Dh;
    float s = 0.0f;
    for (int d = 0; d < Dh; ++d) s += o[d] * o[d];
    atomic_add_float(&score[b * Nq + i], s);
}

// score = alpha * self_imp + beta * ca_score
__kernel void mix_scores(
    __global const float *self_imp,
    __global const float *ca_score,
    __global float *out,
    const int n,
    const float alpha,
    const float beta
) {
    const int i = get_global_id(0);
    if (i >= n) return;
    out[i] = alpha * self_imp[i] + beta * ca_score[i];
}

// out *= (1 + strength * recency[j])
__kernel void apply_recency(
    __global float *scores,
    __global const float *recency,
    const int B,
    const int N,
    const float strength
) {
    const int gid = get_global_id(0);
    if (gid >= B * N) return;
    const int j = gid % N;
    scores[gid] *= (1.0f + strength * recency[j]);
}
