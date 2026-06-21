/* clang-format off */
/*
 * MIT License
 *
 * Copyright (c) 2026 Robyn Kirkman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/* clang-format on */

/*
 * dv_codesearch - selects, per (K, rate) family, the FIVE generator-polynomial
 * sets behind dv_standard_code so that all five are mutually distinguishable: a
 * decoder built for one preset must not lock onto a sibling preset's stream
 * (the property tested by test_cross_lock_within_family / test_lock_matches_
 * compare). It exists so those polynomials are reproducible rather than picked
 * by hand - re-run it to regenerate or re-verify the tables in src/encode.c.
 *
 * Pipeline, per family:
 *   1. Enumerate candidate generator tuples (strictly increasing masks, the
 *      top tap set so the code truly has memory K-1).
 *   2. Keep the non-catastrophic ones (GF(2) gcd of the generators is 1) and
 *      compute each one's free distance (a shortest-path search on the trellis).
 *   3. Take the highest-free-distance `pool` candidates and measure, for every
 *      ordered pair, the decoder lock probability of one code's stream under the
 *      other's decoder - the SAME metric the tests use (decoder_lock_mean: depth
 *      48, max_drift 4, the deterministic test message plus random ones, mean
 *      lock over the settled second half). This O(pool^2) matrix is the costly
 *      step and is fanned out across cores with OpenMP.
 *   4. Pick five codes whose every pairwise cross-lock sits below a separation
 *      target, maximizing the weakest free distance; relax the target only if a
 *      family's code space is too small to meet it (e.g. K3 rate-1/2).
 *
 * Output is a human-readable report plus paste-ready octal tables and the worst
 * pairwise cross-lock achieved per family (which sets each family's test
 * threshold). Like dv_metrics, each lock measurement owns a seeded PRNG stream,
 * so a given seed reproduces exactly regardless of thread count.
 *
 * Usage: dv_codesearch [trials] [info_bits] [seed] [pool]
 */

#include <drift_viterbi/drift_viterbi.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_N 5       /* widest rate this tool handles (1/5)          */
#define MAX_PRESETS 5 /* most presets a family ships (default + 4 alt) */

/* -- deterministic PRNG (splitmix64), matching bench/dv_metrics.c ---------- */

static uint64_t rng_next(uint64_t *state) {
  uint64_t value = (*state += 0x9E3779B97F4A7C15ULL);
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

static void *xmalloc(size_t size) {
  void *ptr = malloc(size);
  if (!ptr) {
    fprintf(stderr, "dv_codesearch: out of memory\n");
    exit(1);
  }
  return ptr;
}

static double max_double(double a, double b) { return a > b ? a : b; }
static double min_double(double a, double b) { return a < b ? a : b; }

/* -- code structure (computed locally, decoupled from the opaque dv_code) --- */

/* GF(2) polynomial gcd of the generators; the code is non-catastrophic iff this
 * is 1 (the Massey-Sain condition for a rate-1/n code, restricted to the
 * canonical no-shared-delay form). Generators are bit masks: bit i is the x^i
 * coefficient. */
static unsigned int gf2_gcd(unsigned int a, unsigned int b) {
  while (b) {
    int da = 31, db = 31;
    while (da >= 0 && !((a >> da) & 1u)) --da;
    while (db >= 0 && !((b >> db) & 1u)) --db;
    if (da < db) {
      unsigned int t = a;
      a = b;
      b = t;
      continue;
    }
    a ^= b << (da - db);
  }
  return a;
}

static int is_noncatastrophic(const unsigned int *gens, int n) {
  unsigned int g = gens[0];
  for (int i = 1; i < n; ++i) {
    g = gf2_gcd(g, gens[i]);
  }
  return g == 1u;
}

/* Output Hamming weight of the n coded bits for (state, input bit), and the next
 * state - the same trellis encode.c builds, recomputed here from generators. */
static int branch_weight(int K, const unsigned int *gens, int n, int state,
                         int bit) {
  unsigned int shift_register = ((unsigned int)bit << (K - 1)) | (unsigned int)state;
  int weight = 0;
  for (int j = 0; j < n; ++j) {
    weight += __builtin_parity(shift_register & gens[j]);
  }
  return weight;
}

static int next_state(int K, int state, int bit) {
  const int n_states = 1 << (K - 1);
  return ((state >> 1) | (bit << (K - 2))) & (n_states - 1);
}

/* Free distance: minimum Hamming weight of a nonzero codeword, i.e. the lowest
 * weight path that leaves the zero state on input 1 and first returns to it.
 * Dijkstra over the n_states trellis with the zero-state self-loop removed; the
 * state space is small (<= 64), so a plain O(V^2) scan suffices. */
static int free_distance(int K, const unsigned int *gens, int n) {
  const int n_states = 1 << (K - 1);
  int dist[1 << 8]; /* K <= 9 -> n_states <= 256; stays on the stack */
  char visited[1 << 8];
  const int INF = 1 << 28;
  for (int s = 0; s < n_states; ++s) {
    dist[s] = INF;
    visited[s] = 0;
  }

  /* Seed from state 0, input 1 (input 0 is the excluded zero self-loop). */
  const int seed_state = next_state(K, 0, 1);
  dist[seed_state] = branch_weight(K, gens, n, 0, 1);
  int best_return = INF;

  for (;;) {
    int u = -1, best = INF;
    for (int s = 1; s < n_states; ++s) { /* state 0 is terminal, never expanded */
      if (!visited[s] && dist[s] < best) {
        best = dist[s];
        u = s;
      }
    }
    if (u < 0) {
      break;
    }
    visited[u] = 1;
    for (int bit = 0; bit <= 1; ++bit) {
      const int v = next_state(K, u, bit);
      const int w = dist[u] + branch_weight(K, gens, n, u, bit);
      if (v == 0) {
        if (w < best_return) best_return = w;
      } else if (!visited[v] && w < dist[v]) {
        dist[v] = w;
      }
    }
  }

  return best_return >= INF ? 0 : best_return;
}

/* -- candidate enumeration ------------------------------------------------- */

typedef struct {
  unsigned int g[MAX_N];
  int dfree;
} candidate;

typedef struct {
  candidate *data;
  size_t len, cap;
} cand_vec;

static void cand_push(cand_vec *v, const unsigned int *g, int n) {
  if (v->len == v->cap) {
    v->cap = v->cap ? v->cap * 2 : 1024;
    v->data = realloc(v->data, v->cap * sizeof(*v->data));
    if (!v->data) {
      fprintf(stderr, "dv_codesearch: out of memory\n");
      exit(1);
    }
  }
  candidate *c = &v->data[v->len++];
  for (int i = 0; i < n; ++i) c->g[i] = g[i];
  for (int i = n; i < MAX_N; ++i) c->g[i] = 0;
  c->dfree = 0;
}

/* Recursively emit every strictly-increasing n-tuple of K-bit generator masks
 * whose largest member has the top (x^{K-1}) tap set, so the encoder genuinely
 * uses K-1 memory bits. Catastrophic / low-distance tuples are pruned later. */
static void enumerate(cand_vec *v, int K, int n, int depth, unsigned int low,
                      unsigned int *acc) {
  const unsigned int top = 1u << (K - 1);
  const unsigned int hi = (1u << K) - 1u;
  if (depth == n) {
    if (acc[n - 1] & top) { /* largest mask carries the top tap */
      cand_push(v, acc, n);
    }
    return;
  }
  /* Leave room for the remaining (n - depth) strictly-greater generators. */
  const unsigned int last = hi - (unsigned int)(n - 1 - depth);
  for (unsigned int g = low; g <= last; ++g) {
    acc[depth] = g;
    enumerate(v, K, n, depth + 1, g + 1, acc);
  }
}

static int cand_cmp(const void *pa, const void *pb) {
  const candidate *a = pa, *b = pb;
  if (a->dfree != b->dfree) return b->dfree - a->dfree; /* higher d_free first */
  for (int i = 0; i < MAX_N; ++i) {                     /* deterministic tie-break */
    if (a->g[i] != b->g[i]) return (a->g[i] < b->g[i]) ? -1 : 1;
  }
  return 0;
}

/* -- lock metric (mirrors tests/dv_test_util.h decoder_lock_mean) ----------- */

/* Mean lock probability over the settled second half when `enc`'s coded stream
 * is decoded with `dec`'s decoder. Same settings the test helper uses. */
static double lock_mean(const dv_code *enc, const dv_code *dec,
                        const uint8_t *msg, int info_bits, int depth) {
  const int clen = info_bits * dv_code_n(enc);
  uint8_t *coded = xmalloc((size_t)clen);
  int st = 0;
  dv_code_encode(enc, msg, info_bits, &st, coded);

  dv_stream_params params = {.decision_depth = depth,
                             .max_drift = 4,
                             .p_sub = 0.01,
                             .p_ins = 0.01,
                             .p_del = 0.01,
                             .p_erase = 0.0};
  dv_stream_decoder *sd = dv_stream_decoder_create(dec, &params);
  if (!sd) {
    free(coded);
    return 1.0; /* treat as worst case (indistinguishable) */
  }
  const int cap = info_bits + 64;
  uint8_t *out = xmalloc((size_t)cap);
  double *lock = xmalloc((size_t)cap * sizeof(double));
  int got = dv_stream_decode(sd, coded, clen, out, lock, cap);

  double result = 1.0;
  if (got > 0) {
    double sum = 0.0;
    int count = 0;
    for (int i = got / 2; i < got; ++i) {
      sum += lock[i];
      ++count;
    }
    result = count ? sum / count : 1.0;
  }
  dv_stream_decoder_destroy(sd);
  free(coded);
  free(out);
  free(lock);
  return result;
}

/* Aggregate lock_mean over every test message: the conservative direction so a
 * single lucky message can't inflate distinguishability. For self pairs we take
 * the MIN observed lock (the worst self-lock); for cross pairs the MAX (the
 * worst cross-lock). */
static double lock_agg(const dv_code *enc, const dv_code *dec,
                       const uint8_t *messages, int n_msg, int info_bits,
                       int depth, int take_min) {
  double agg = take_min ? 1.0 : 0.0;
  for (int m = 0; m < n_msg; ++m) {
    double v = lock_mean(enc, dec, messages + (size_t)m * info_bits, info_bits,
                         depth);
    agg = take_min ? min_double(agg, v) : max_double(agg, v);
  }
  return agg;
}

/* Blind dv_compare similarity between two codes' clean streams (~1 same, ~0
 * different). The distinguishability tests cross-check lock against dv_compare,
 * so the selected five must separate under this metric too; this verifies the
 * final pick rather than driving it (dv_compare is the costlier metric). */
static double compare_codes(const dv_code *a, const dv_code *b,
                            const uint8_t *msg, int info_bits) {
  const int n = dv_code_n(a), K = dv_code_k(a);
  const int ca = info_bits * n, cb = info_bits * dv_code_n(b);
  uint8_t *sa = xmalloc((size_t)ca);
  uint8_t *sb = xmalloc((size_t)cb);
  int st = 0;
  dv_code_encode(a, msg, info_bits, &st, sa);
  st = 0;
  dv_code_encode(b, msg, info_bits, &st, sb);
  double result = dv_compare(n, K, sa, (size_t)ca, sb, (size_t)cb);
  free(sa);
  free(sb);
  return result;
}

/* -- selection ------------------------------------------------------------- */

/* -- families -------------------------------------------------------------- */

typedef struct {
  const char *name;
  const char *enum_prefix; /* e.g. DV_CODE_K7_RATE_1_2 */
  int K, n;
  /* Alternate-selection margin. 0: distinguishable at the normal 0.75 bound
   * (strongest codes that stay distinguishable). 1: hold the alternates to a
   * tighter cross-lock bound (a wider margin under the test's 0.75), trading a
   * little free distance for that headroom. */
  int wide;
  /* Optional pinned default: force this exact generator set to be the family's
   * first (default) code, with the alternates selected around it. pin_n == 0
   * leaves the default to the search (the strongest distinguishable code). Used
   * to keep a canonical, widely-recognized polynomial as the default. */
  int pin_n;
  unsigned int pin[MAX_N];
} family;

static const family FAMILIES[] = {
    {"K3_R1_2", "DV_CODE_K3_RATE_1_2", 3, 2, 0, 0, {0}},
    /* Pin the canonical NASA/Voyager K=7 rate-1/2 code (0171, 0133) as default. */
    {"K7_R1_2", "DV_CODE_K7_RATE_1_2", 7, 2, 0, 2, {0171, 0133}},
    /* Rate-1/3 has a roomy distinguishable set, so spread the five out for a
     * wider cross-lock margin rather than squeezing maximum free distance. */
    {"K7_R1_3", "DV_CODE_K7_RATE_1_3", 7, 3, 1, 0, {0}},
    {"K5_R1_5", "DV_CODE_K5_RATE_1_5", 5, 5, 0, 0, {0}},
};
#define N_FAMILIES ((int)(sizeof(FAMILIES) / sizeof(FAMILIES[0])))

/* Minimum acceptable self-lock for a pool member (matches the test's > 0.9). */
static const double SELF_MIN = 0.9;

static void run_family(const family *fam, const uint8_t *messages, int n_msg,
                       int info_bits, int depth, int sample_size,
                       uint64_t sample_seed) {
  printf("=== %s (K=%d, n=%d) ===\n", fam->name, fam->K, fam->n);

  /* 1. Enumerate, then compute d_free and reject catastrophic codes. */
  cand_vec all = {0};
  unsigned int acc[MAX_N];
  enumerate(&all, fam->K, fam->n, 0, 1u, acc);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
  for (long i = 0; i < (long)all.len; ++i) {
    candidate *c = &all.data[i];
    c->dfree = is_noncatastrophic(c->g, fam->n)
                   ? free_distance(fam->K, c->g, fam->n)
                   : 0; /* 0 marks "drop me" */
  }

  /* Compact to the viable candidates, then rank by free distance. */
  size_t viable = 0;
  for (size_t i = 0; i < all.len; ++i) {
    if (all.data[i].dfree > 0) all.data[viable++] = all.data[i];
  }
  qsort(all.data, viable, sizeof(*all.data), cand_cmp);
  printf("  enumerated %zu tuples, %zu non-catastrophic\n", all.len, viable);

  int nsamp = sample_size < (int)viable ? sample_size : (int)viable;
  if (nsamp < 5) {
    printf("  ERROR: only %d viable codes, need >= 5\n", nsamp);
    free(all.data);
    return;
  }

  /* Build the candidate sample. The very highest-free-distance codes form a
   * tight, codeword-equivalent cluster that mutually lock (e.g. for rate-1/2 K7
   * the optimal d_free-10 codes all lock onto one another), so picking the five
   * by free distance alone fails - at most one cluster member is usable. We keep
   * the top few for strength, then draw the rest at RANDOM from the whole viable
   * set: most codes are well separated, so a broad random sample reliably
   * contains a mutually-distinguishable five. */
  /* Force in only the single best code (a strong default); the optimal codes are
   * a mutually-locking cluster, so forcing in several merely crowds out the
   * separated codes we need. Everything else is drawn at random. */
  const int keep_top = 1;
  candidate *samp = xmalloc((size_t)nsamp * sizeof(*samp));
  for (int i = 0; i < keep_top && i < nsamp; ++i) samp[i] = all.data[i];

  /* If this family pins its default, force that exact code into the sample (as
   * the strong slot) so the selection can seed from it. */
  if (fam->pin_n == fam->n) {
    candidate pc;
    for (int j = 0; j < fam->n; ++j) pc.g[j] = fam->pin[j];
    for (int j = fam->n; j < MAX_N; ++j) pc.g[j] = 0;
    pc.dfree = free_distance(fam->K, pc.g, fam->n);
    samp[0] = pc;
  }

  /* Partial Fisher-Yates over ranks [keep_top, viable) for the random slots. */
  size_t *order = xmalloc((viable - (size_t)keep_top) * sizeof(*order));
  for (size_t i = 0; i < viable - (size_t)keep_top; ++i)
    order[i] = (size_t)keep_top + i;
  uint64_t rng = sample_seed ^ ((uint64_t)fam->K << 8) ^ (uint64_t)fam->n;
  for (int i = keep_top; i < nsamp; ++i) {
    const size_t span = viable - (size_t)i; /* unpicked entries remain in [i-kt..) */
    const size_t pick = (size_t)(i - keep_top) + rng_next(&rng) % span;
    const size_t tmp = order[i - keep_top];
    order[i - keep_top] = order[pick];
    order[pick] = tmp;
    samp[i] = all.data[order[i - keep_top]];
  }
  free(order);

  /* Sort the sample by free distance (descending) so the farthest-point seed is
   * the strongest code and ties favour higher d_free. */
  qsort(samp, (size_t)nsamp, sizeof(*samp), cand_cmp);

  /* 2. Build a decoder-ready code for each sample member (shared read-only). */
  dv_code **codes = xmalloc((size_t)nsamp * sizeof(*codes));
  for (int i = 0; i < nsamp; ++i) {
    codes[i] = dv_code_create(fam->K, samp[i].g, fam->n);
    if (!codes[i]) {
      fprintf(stderr, "dv_codesearch: code create failed\n");
      exit(1);
    }
  }

  /* 3. Self-lock of every sample member (parallel). A code that doesn't lock
   *    onto its own stream can't be a preset. */
  double *self = xmalloc((size_t)nsamp * sizeof(double));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 0; i < nsamp; ++i) {
    self[i] = lock_agg(codes[i], codes[i], messages, n_msg, info_bits, depth,
                       /*take_min=*/1);
  }

  /* 4. Full pairwise cross-lock matrix over the sample (parallel). */
  double *M = xmalloc((size_t)nsamp * nsamp * sizeof(double));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int idx = 0; idx < nsamp * nsamp; ++idx) {
    const int i = idx / nsamp, j = idx % nsamp;
    M[idx] = (i == j) ? self[i]
                      : lock_agg(codes[i], codes[j], messages, n_msg, info_bits,
                                 depth, /*take_min=*/0);
  }
  double *pair_cross = xmalloc((size_t)nsamp * nsamp * sizeof(double));
  for (int i = 0; i < nsamp; ++i)
    for (int j = 0; j < nsamp; ++j)
      pair_cross[(size_t)i * nsamp + j] =
          max_double(M[(size_t)i * nsamp + j], M[(size_t)j * nsamp + i]);

  /* Largest mutually-distinguishable set at the test bound, by multi-start
   * greedy: from each possible seed, grow a clique by adding (in d_free order)
   * every code that stays below DISTINCT_BOUND against all already kept, and
   * keep the largest clique found. Its size is how many mutually-distinguishable
   * presets this family supports - a tight lower bound on the true ceiling.
   * Rate-1/2 families top out around 3 here; that's expected, and we simply ship
   * the best that-many rather than padding to five with codes that aren't
   * actually distinguishable. */
  const double DISTINCT_BOUND = 0.75;
  int *set = xmalloc((size_t)nsamp * sizeof(int));
  int nset = 0; /* how many mutually-distinguishable codes the family supports */
  for (int seed = 0; seed < nsamp; ++seed) {
    if (self[seed] < SELF_MIN) continue;
    int n = 0;
    int *s = set;
    s[n++] = seed;
    for (int c = 0; c < nsamp; ++c) {
      if (c == seed || self[c] < SELF_MIN) continue;
      int ok = 1;
      for (int t = 0; t < n; ++t)
        if (pair_cross[(size_t)s[t] * nsamp + c] >= DISTINCT_BOUND) {
          ok = 0;
          break;
        }
      if (ok) s[n++] = c;
    }
    if (n > nset) nset = n;
  }
  free(set);

  /* Grow the preset set one code at a time from a seed (the pinned default if
   * the family pins one, else the strongest self-locking code). Each round adds
   * the highest-free-distance candidate that still stays distinguishable from all
   * already chosen - i.e. worst cross-lock below the family's selection bound:
   * the normal 0.75, or a tighter SELECT_BOUND_WIDE for families that want extra
   * margin. Stops at MAX_PRESETS or when nothing stays under the bound, so
   * rate-1/2 naturally stops at its ~3-code ceiling. */
  const double SELECT_BOUND_WIDE = 0.65;
  const double select_bound = fam->wide ? SELECT_BOUND_WIDE : DISTINCT_BOUND;
  int seed = -1;
  if (fam->pin_n == fam->n) {
    for (int c = 0; c < nsamp; ++c) {
      int match = 1;
      for (int j = 0; j < fam->n; ++j)
        if (samp[c].g[j] != fam->pin[j]) {
          match = 0;
          break;
        }
      if (match) {
        seed = c;
        break;
      }
    }
  }
  if (seed < 0) { /* highest-d_free self-locking code (samp is d_free-desc) */
    seed = 0;
    while (seed < nsamp && self[seed] < SELF_MIN) ++seed;
    if (seed >= nsamp) seed = 0;
  }

  const int target = nset < MAX_PRESETS ? nset : MAX_PRESETS;
  int sel[MAX_PRESETS];
  int count = 0;
  char *chosen = xmalloc((size_t)nsamp);
  memset(chosen, 0, (size_t)nsamp);
  double *worst = xmalloc((size_t)nsamp * sizeof(double));
  sel[count++] = seed;
  chosen[seed] = 1;
  for (int c = 0; c < nsamp; ++c)
    worst[c] = pair_cross[(size_t)seed * nsamp + c];
  while (count < target) {
    int best = -1;
    for (int c = 0; c < nsamp; ++c) { /* samp is d_free-desc: first eligible wins */
      if (chosen[c] || self[c] < SELF_MIN) continue;
      if (worst[c] >= select_bound) continue; /* not distinguishable enough */
      best = c;
      break;
    }
    if (best < 0) break;
    sel[count++] = best;
    chosen[best] = 1;
    for (int c = 0; c < nsamp; ++c)
      worst[c] = max_double(worst[c], pair_cross[(size_t)best * nsamp + c]);
  }
  free(chosen);
  free(worst);

  /* Keep the default (sel[0] = pin or strongest) first; order the alternates by
   * free distance descending. */
  for (int i = 2; i < count; ++i) {
    int v = sel[i], j = i - 1;
    while (j >= 1 && samp[sel[j]].dfree < samp[v].dfree) {
      sel[j + 1] = sel[j];
      --j;
    }
    sel[j + 1] = v;
  }
  printf("  distinguishable set at %.2f: %d codes (shipping %d)\n",
         DISTINCT_BOUND, nset, count);

  double M5[MAX_PRESETS][MAX_PRESETS];
  double worst_cross = 0.0;
  for (int i = 0; i < count; ++i)
    for (int j = 0; j < count; ++j) {
      M5[i][j] = M[(size_t)sel[i] * nsamp + sel[j]];
      if (i != j) worst_cross = max_double(worst_cross, M5[i][j]);
    }
  free(M);
  free(pair_cross);

  printf("  selected %d (worst pairwise cross-lock %.3f, weakest d_free %d):\n",
         count, worst_cross, samp[sel[count - 1]].dfree);
  for (int i = 0; i < count; ++i) {
    const candidate *c = &samp[sel[i]];
    printf("    [%d] {", i);
    for (int j = 0; j < fam->n; ++j) printf("%s0%o", j ? ", " : "", c->g[j]);
    printf("}  d_free=%d  self_lock=%.3f\n", c->dfree, self[sel[i]]);
  }

  /* Pairwise cross-lock among the selected codes (the numbers behind the test
   * threshold for this family). */
  printf("  cross-lock matrix (selected, row=stream / col=decoder):\n     ");
  for (int j = 0; j < count; ++j) printf("   [%d] ", j);
  printf("\n");
  for (int i = 0; i < count; ++i) {
    printf("    [%d]", i);
    for (int j = 0; j < count; ++j) printf(" %.3f", M5[i][j]);
    printf("\n");
  }

  /* Paste-ready encoder cases. The first (highest-d_free) becomes the family
   * default; the rest its ALT1..ALT(count-1). */
  static const char *suffix[5] = {"", "_ALT1", "_ALT2", "_ALT3", "_ALT4"};
  printf("  --- encode.c cases ---\n");
  for (int i = 0; i < count; ++i) {
    const candidate *c = &samp[sel[i]];
    printf("    case %s%s: {\n", fam->enum_prefix, suffix[i]);
    printf("      static const unsigned int generators[] = {");
    for (int j = 0; j < fam->n; ++j) printf("%s0%o", j ? ", " : "", c->g[j]);
    printf("};\n      return dv_code_create(%d, generators, %d);\n    }\n",
           fam->K, fam->n);
  }
  printf("  --- encode.h d_free comments ---\n");
  for (int i = 0; i < count; ++i) {
    printf("    %s%-5s /* d_free %d */\n", fam->enum_prefix, suffix[i],
           samp[sel[i]].dfree);
  }

  /* Cross-check the pick under dv_compare (the other route the tests use):
   * every off-diagonal pair must read as different, the diagonal as same.
   * dv_compare needs a longer stream than lock selection to recover the dual
   * space, so it gets its own dedicated message - independent of the (short)
   * selection length. */
  const int verify_bits = 2000;
  uint8_t *vmsg = xmalloc((size_t)verify_bits);
  uint64_t vrng = sample_seed ^ 0x5151515151515151ULL;
  for (int i = 0; i < verify_bits; ++i) vmsg[i] = (uint8_t)(rng_next(&vrng) & 1u);
  double worst_compare_cross = 0.0, worst_compare_self = 1.0;
  for (int i = 0; i < count; ++i) {
    worst_compare_self = min_double(
        worst_compare_self,
        compare_codes(codes[sel[i]], codes[sel[i]], vmsg, verify_bits));
    for (int j = i + 1; j < count; ++j)
      worst_compare_cross = max_double(
          worst_compare_cross,
          compare_codes(codes[sel[i]], codes[sel[j]], vmsg, verify_bits));
  }
  free(vmsg);
  printf("  dv_compare check: worst self=%.3f, worst cross=%.3f %s\n",
         worst_compare_self, worst_compare_cross,
         (worst_compare_self > 0.5 && worst_compare_cross < 0.5) ? "OK"
                                                                 : "FAIL");

  for (int i = 0; i < nsamp; ++i) dv_code_destroy(codes[i]);
  free(codes);
  free(self);
  free(samp);
  free(all.data);
  printf("\n");
  fflush(stdout);
}

int main(int argc, char **argv) {
  int trials = argc > 1 ? atoi(argv[1]) : 8;
  int info_bits = argc > 2 ? atoi(argv[2]) : 600;
  uint64_t seed = argc > 3 ? strtoull(argv[3], NULL, 0) : 0xC0FFEEULL;
  int sample_size = argc > 4 ? atoi(argv[4]) : 256;
  int only_family = argc > 5 ? atoi(argv[5]) : -1; /* -1 = all families */
  if (trials < 1 || info_bits < 64 || sample_size < 5 ||
      only_family >= N_FAMILIES) {
    fprintf(stderr,
            "usage: %s [trials>=1] [info_bits>=64] [seed] [sample>=5] "
            "[family 0..%d, -1=all]\n",
            argv[0], N_FAMILIES - 1);
    return 2;
  }
  const int depth = 48; /* the decision depth the distinguishability tests use */

  /* Shared message set: message 0 is the deterministic message the tests use,
   * so a selection that clears our target also clears the test; the rest are
   * random, broadening the worst-case aggregate. */
  uint8_t *messages = xmalloc((size_t)trials * info_bits);
  for (int i = 0; i < info_bits; ++i)
    messages[i] = (uint8_t)((i * 7 + 3) & 1);
  uint64_t rng = seed;
  for (int m = 1; m < trials; ++m) {
    for (int i = 0; i < info_bits; ++i)
      messages[(size_t)m * info_bits + i] = (uint8_t)(rng_next(&rng) & 1u);
  }

#ifdef _OPENMP
  fprintf(stderr, "dv_codesearch: %d threads, %d messages x %d info bits, "
                  "sample %d, seed 0x%llx\n",
          omp_get_max_threads(), trials, info_bits, sample_size,
          (unsigned long long)seed);
#else
  fprintf(stderr, "dv_codesearch: single-threaded, %d messages x %d info bits, "
                  "sample %d, seed 0x%llx\n",
          trials, info_bits, sample_size, (unsigned long long)seed);
#endif

  for (int f = 0; f < N_FAMILIES; ++f) {
    if (only_family >= 0 && f != only_family) continue;
    run_family(&FAMILIES[f], messages, trials, info_bits, depth, sample_size,
               seed);
  }

  free(messages);
  return 0;
}
