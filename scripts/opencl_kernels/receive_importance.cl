// JEPA postfuse attention-receive importance (OpenCL)
// Matches train_stream_mtp_concat_ca._postfuse_attn_importance:
//   x = L2-normalize(tokens)
//   for each query chunk C:
//     logits[b,c,j] = scale * dot(q[c], k[j])
//     attn = softmax_j(logits)
//     imp[b,j] += sum_c attn[b,c,j]

__kernel void l2_normalize_rows(
    __global const float *in,   // [B, N, D]
    __global float *out,        // [B, N, D]
    const int rows,             // B*N
    const int D
) {
    const int r = get_global_id(0);
    if (r >= rows) return;
    __global const float *src = in + r * D;
    __global float *dst = out + r * D;
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += src[d] * src[d];
    float inv = rsqrt(fmax(s, 1e-12f));
    for (int d = 0; d < D; ++d) dst[d] = src[d] * inv;
}

__kernel void qk_logits_chunk(
    __global const float *x,      // [B, N, D] normalized
    __global float *logits,       // [B, C, N]
    const int B,
    const int N,
    const int D,
    const int q_start,
    const int C,
    const float scale
) {
    const size_t gid = get_global_id(0);
    const size_t total = (size_t)B * (size_t)C * (size_t)N;
    if (gid >= total) return;

    const int j = (int)(gid % (size_t)N);
    const int tmp = (int)(gid / (size_t)N);
    const int c = tmp % C;
    const int b = tmp / C;
    const int i = q_start + c;

    __global const float *q = x + ((b * N + i) * D);
    __global const float *k = x + ((b * N + j) * D);
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += q[d] * k[d];
    }
    logits[gid] = dot * scale;
}

__kernel void softmax_rows(
    __global const float *logits,  // [rows, N]
    __global float *attn,          // [rows, N]
    const int rows,
    const int N
) {
    const int row = get_global_id(0);
    if (row >= rows) return;
    __global const float *src = logits + (size_t)row * (size_t)N;
    __global float *dst = attn + (size_t)row * (size_t)N;

    float m = -INFINITY;
    for (int j = 0; j < N; ++j) m = fmax(m, src[j]);
    float z = 0.0f;
    for (int j = 0; j < N; ++j) {
        float e = exp(src[j] - m);
        dst[j] = e;
        z += e;
    }
    float inv_z = 1.0f / z;
    for (int j = 0; j < N; ++j) dst[j] *= inv_z;
}

__kernel void reduce_chunk_to_imp(
    __global const float *attn,  // [B, C, N]
    __global float *imp,         // [B, N]
    const int B,
    const int N,
    const int C
) {
    const int gid = get_global_id(0);  // b*N + j
    const int total = B * N;
    if (gid >= total) return;
    const int b = gid / N;
    const int j = gid - b * N;
    float s = 0.0f;
    for (int c = 0; c < C; ++c) {
        s += attn[((b * C + c) * N) + j];
    }
    imp[gid] += s;
}
