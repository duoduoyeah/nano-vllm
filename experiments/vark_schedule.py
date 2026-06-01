"""Balanced ragged-K schedule for the variable-K (S=1) decode experiment.

build_balanced_schedule(batch, output_len, k_min, k_max, num_steps, seed) -> B x num_steps matrix M,
M[i][t] = tokens request i commits at step t, with BOTH:
  * every ROW sums to output_len   -> each request emits exactly output_len tokens, AND
  * every COLUMN sums to the same constant total_q = batch*output_len/num_steps
    -> constant query rows per step, so a SINGLE varlen cudagraph covers every step.
Entries lie in [k_min, k_max]. Built from the uniform `avg` matrix by random sum-preserving 2x2
rotations (+1 at (r1,c1),(r2,c2); -1 at (r1,c2),(r2,c1)), which leave every row and column sum
unchanged. Deterministic given `seed`. With output_len=256, num_steps=64 -> avg=4, range [1,8].
"""
import random


def build_balanced_schedule(batch, output_len=256, k_min=1, k_max=8, num_steps=64, seed=0):
    assert output_len % num_steps == 0, "output_len must be divisible by num_steps"
    avg = output_len // num_steps
    assert k_min <= avg <= k_max, f"avg per step {avg} must lie in [{k_min}, {k_max}]"
    M = [[avg] * num_steps for _ in range(batch)]   # rows sum to output_len, cols sum to batch*avg
    rng = random.Random(seed)
    # Each accepted rotation moves one unit while preserving all row/column sums; many rotations
    # spread the entries across [k_min, k_max] without ever breaking the balance.
    for _ in range(batch * num_steps * 8):
        r1, r2 = rng.randrange(batch), rng.randrange(batch)
        c1, c2 = rng.randrange(num_steps), rng.randrange(num_steps)
        if r1 == r2 or c1 == c2:
            continue
        if M[r1][c1] < k_max and M[r2][c2] < k_max and M[r1][c2] > k_min and M[r2][c1] > k_min:
            M[r1][c1] += 1; M[r2][c2] += 1; M[r1][c2] -= 1; M[r2][c1] -= 1
    # invariants the engine relies on (constant total_q per step; each request emits output_len)
    assert all(sum(row) == output_len for row in M)
    col_total = batch * output_len // num_steps
    assert all(sum(M[i][t] for i in range(batch)) == col_total for t in range(num_steps))
    return M


def describe(M):
    """Return (min, max, mean, distinct per-step totals) for logging the realized raggedness."""
    flat = [x for row in M for x in row]
    col_totals = {sum(M[i][t] for i in range(len(M))) for t in range(len(M[0]))}
    return min(flat), max(flat), sum(flat) / len(flat), sorted(col_totals)


if __name__ == "__main__":
    M = build_balanced_schedule(64, 256, 1, 8, 64, 0)
    lo, hi, mean, totals = describe(M)
    print(f"64x64 schedule: K in [{lo},{hi}], mean {mean:.3f}, per-step totals {totals} (want one value)")
