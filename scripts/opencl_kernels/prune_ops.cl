// Token prune helpers after fusion (postfuse importance already in receive_importance.cl)

__kernel void recency_reweight(
    __global float *scores,          // [B, N] in/out
    __global const float *recency,   // [N] in (0,1]
    const int B,
    const int N,
    const float strength
) {
    const int gid = get_global_id(0);
    const int total = B * N;
    if (gid >= total) return;
    const int j = gid % N;
    scores[gid] *= (1.0f + strength * recency[j]);
}

// Fill recency[j] = (slot+1)/n_slots with slot = j / gp
__kernel void fill_recency(
    __global float *recency,
    const int N,
    const int gp
) {
    const int j = get_global_id(0);
    if (j >= N) return;
    int g = gp < 1 ? 1 : gp;
    int n_slots = (N + g - 1) / g;
    if (n_slots < 1) n_slots = 1;
    int slot = j / g;
    recency[j] = ((float)slot + 1.0f) / (float)n_slots;
}

// Gather tokens: out[b,k,d] = in[b, idx[b,k], d]
__kernel void gather_tokens(
    __global const float *in,   // [B, N, D]
    __global const int *idx,    // [B, K]
    __global float *out,        // [B, K, D]
    const int B,
    const int N,
    const int K,
    const int D
) {
    const size_t gid = get_global_id(0);
    const size_t total = (size_t)B * (size_t)K * (size_t)D;
    if (gid >= total) return;
    const int d = (int)(gid % (size_t)D);
    const int tmp = (int)(gid / (size_t)D);
    const int k = tmp % K;
    const int b = tmp / K;
    const int j = idx[b * K + k];
    out[gid] = in[((b * N + j) * D) + d];
}

// Arg-topk (partial): each WI owns one batch row; selection-sort style for small N.
// Writes unsorted top-K indices into idx[B,K] (host may sort for keep-order).
__kernel void topk_indices_row(
    __global const float *scores,  // [B, N]
    __global int *idx,             // [B, K]
    const int B,
    const int N,
    const int K
) {
    const int b = get_global_id(0);
    if (b >= B) return;
    __global const float *s = scores + b * N;
    __global int *o = idx + b * K;

    // Mark selected with a scratch of -inf by copying scores to private? Use O(NK) scan.
    // For prototype: maintain taken bitmap in local... N may be large.
    // Simple O(K*N) without bitmap: for each k find max among not-yet-picked by
    // checking against previously chosen indices.
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int best_j = 0;
        for (int j = 0; j < N; ++j) {
            int taken = 0;
            for (int t = 0; t < k; ++t) {
                if (o[t] == j) { taken = 1; break; }
            }
            if (taken) continue;
            if (s[j] > best) {
                best = s[j];
                best_j = j;
            }
        }
        o[k] = best_j;
    }
}
