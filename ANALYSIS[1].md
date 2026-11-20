# ANALYSIS — Cache Line & False Sharing Mechanics

Host: `<cpu_model>` · logical cores: `<N>` · cache line: `<64>` B · threads: 4 · iterations/thread: 60,000,000
(Всё, что в угловых скобках `<...>`, замените числами из вашего `cache_lab_manifest.json`.)

## Measured results (from manifest)

| Layout | Stride | Elapsed (ns) | ns / update / thread | Ctx switches (vol/invol) |
|---|---|---|---|---|
| A Contiguous | 1 (8 B) | `<...>` | `<...>` | `<...>` |
| B Half-line | 4 (32 B) | `<...>` | `<...>` | `<...>` |
| C Line-padded | 8 (64 B) | `<...>` | `<...>` | `<...>` |
| D Prefetch-safe | 16 (128 B) | `<...>` | `<...>` | `<...>` |
| Privatized (register) | 8 | `<...>` | `<...>` | `<...>` |

## Q1 — MESI transitions (stride = 1)

Initially the line holding `buffer[0..7]` is in some state, e.g. **E** or **M** in Core 0 after its first store.

1. Core 0 stores `buffer[0]`: it issues an RFO (or already owns the line). All other copies go **Invalid**; Core 0's line becomes **Modified**.
2. Core 1 stores `buffer[1]` (a different address, same 64-byte line). Its L1 copy is **Invalid** (or absent), so it must issue an RFO.
3. Core 0 snoops the RFO, writes back / forwards the Modified line, and transitions **M → I**. Core 1 receives the line and goes **I → M** (the intermediate step is the fill in E-with-intent-to-write, completing as M after the store retires).
4. Core 0's next store to `buffer[0]` finds its line **Invalid** and sends an RFO, taking the line back: **I → M** for Core 0, **M → I** for Core 1. This ping-pong repeats every iteration.

Why Core 1's store buffer stalls: a store cannot commit to L1 until the core holds the line in M/E state. The store waits at the head of the store buffer (x86 TSO commits stores in order), so younger stores queue behind it. When the buffer fills, the pipeline stalls. Each RFO round trip costs tens to hundreds of cycles (cross-core snoop / L3 or interconnect latency), versus ~1 cycle for an L1 hit on a privately owned line. With 4 cores, up to 3 other copies must be invalidated per ownership transfer.

## Q2 — Quantitative slowdown

- stride = 1: `<T1>` ns → `<T1/6e7>` ns/update/thread
- stride = 4: `<T4>` ns → `<...>` ns/update
- stride = 8: `<T8>` ns → `<...>` ns/update
- stride = 16: `<T16>` ns → `<...>` ns/update
- private: `<TP>` ns → `<...>` ns/update

Slowdown stride1 / stride8 = `<T1 / T8>`× (manifest key `slowdown_stride1_vs_stride8`).
Average ns/update = elapsed_ns / 60,000,000 (per thread, all threads run in parallel). Aggregate throughput cost = elapsed / (4 × 60M).

Interpretation: stride 8 vs private baseline isolates the cost of the L1 store→load dependency chain (store-forwarding, ~4–5 cycles per iteration) from coherence traffic. Stride 1 minus stride 8 is the pure cross-core coherence penalty. Stride 4 should sit between them (2-way sharing).

## Q3 — Stride 8 vs stride 16, adjacent-line prefetcher

Measured: stride 8 = `<T8>` ns, stride 16 = `<T16>` ns (ratio `<T8/T16>`).

Many Intel cores have an L2 "adjacent cache line prefetch" (spatial prefetcher) that fetches the buddy line so that 128-byte aligned pairs (even/odd 64-byte lines) are brought together. A core writing line 2k can pull line 2k+1 in (usually Shared/Exclusive), and a peer's RFO on 2k+1 then invalidates/steals that prefetched copy or forces the prefetcher to re-fetch, causing secondary invalidations even though the threads touch different 64-byte lines.

With 64-byte spacing, thread offsets 0, 64, 128, 192 put threads 0&1 in one 128-byte sector and 2&3 in another, so 64-byte padding can still suffer pairing interference when the prefetcher is enabled and the buffer is 128-byte aligned. With 128-byte spacing each thread owns its own sector, which removes it. If the measured difference is small (e.g. AMD Zen, Apple M-series, or prefetcher disabled in BIOS) the effect is not visible; report that honestly with your numbers.

## Q4 — Register promotion vs real stores

An optimizer (LLVM LICM scalar promotion / mem2reg-style promotion) may notice that `buffer[slot]` is loop-invariant in address, loaded and stored only by this loop, and keep it in a register for the entire loop, storing once at the end. That would be legal for single-threaded semantics and would delete all memory traffic (and the false sharing).

In `worker_strided_accumulate` the loop body is `buffer[slot] = buffer[slot] * mult + (s ^ i)`: a loop-carried read-modify-write whose value is an LCG-style chain. In the emitted IR (`<paste the relevant lines from manifest llvm.sample_memory_ops, e.g. 'load i64, i64* %...' and 'store i64 %..., i64* %...'>`) the `load i64` and `store i64` appear inside the loop body (`<basic block label>`). `<State here honestly what your IR shows>`. Note: the manifest check only proves that i64 loads/stores exist in the IR and that there are no `Py*` object calls; confirm the store is inside the loop block, not after it. If LLVM had promoted it, the privatized baseline and strided kernel would show the same time, which your measurements `<do / do not>` show.

## Q5 — GIL vs nogil

Without `nogil=True`, only one thread executes Python bytecode at a time; threads take turns holding the GIL (switch interval ≈ 5 ms). Two threads therefore never store to the same cache line simultaneously, and the stores happen from one core at a time, so no RFO ping-pong occurs: there is no false sharing, only serialization.

Adding threads makes it slower because of **software** contention: GIL acquire/release, condition-variable signaling, OS context switches (visible in `ru_nvcsw`/`ru_nivcsw`) and lost cache locality when the interpreter state moves between cores. This is lock contention in the interpreter, whose cost is at the OS scheduler level (microseconds) and does not depend on data layout (stride 1 and stride 8 time the same). With `nogil=True`, compiled code runs truly in parallel on different cores, so the cost comes from **hardware** cache-line contention (RFO / invalidations, tens to hundreds of cycles) and does depend strongly on stride.
