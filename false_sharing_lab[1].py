import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "REPLACE_WITH_YOUR_GIT_USERNAME"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)
LINE_BYTES = 64


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def read_cpu_model() -> str:
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8"
            ).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_model = read_cpu_model()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    # Deterministic 64-bit seed: SHA-256 over git handle + CPU signature + model string
    digest = hashlib.sha256(f"{git_user}::{cpu_sig}::{cpu_model}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    # Core affinity (Linux only): pick NUM_THREADS distinct cores from the allowed set
    affinity_supported = hasattr(os, "sched_setaffinity")
    core_ids = []
    if affinity_supported:
        allowed = sorted(os.sched_getaffinity(0))
        core_ids = allowed[:NUM_THREADS]
        if len(core_ids) < NUM_THREADS:
            affinity_supported = False
            core_ids = []

    return {
        "git_user": git_user,
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "cpu_model": cpu_model,
        "cpu_sig": cpu_sig,
        "instance_seed": instance_seed,
        "affinity_supported": affinity_supported,
        "core_ids": core_ids,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def make_aligned_buffer(n_elems: int, align: int = 64) -> np.ndarray:
    """uint64 array whose data pointer is aligned to `align` bytes."""
    raw = np.zeros(n_elems * 8 + align, dtype=np.uint8)
    addr = raw.ctypes.data
    offset = (-addr) % align
    buf = raw[offset: offset + n_elems * 8].view(np.uint64)
    assert buf.ctypes.data % align == 0
    return buf


# ==============================================================================
# TASK 2: GIL-free native counter kernel
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    # In-place read-modify-write on buffer[slot] every iteration.
    # The new value depends on the previous memory value (multiplicative chain),
    # so each iteration needs a load -> ALU -> store round trip.
    mult = uint64(6364136223846793005)
    s = uint64(seed)
    for i in range(iters):
        buffer[slot] = buffer[slot] * mult + (s ^ uint64(i))


# ==============================================================================
# TASK 4: Register privatization baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    mult = uint64(6364136223846793005)
    s = uint64(seed)
    acc = buffer[slot]
    for i in range(iters):
        acc = acc * mult + (s ^ uint64(i))
    buffer[slot] = acc  # single shared write


def _thread_target(kernel, buf, slot, iters, seed, core, barrier, times):
    if core is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})  # pid 0 = calling thread on Linux
        except OSError:
            pass
    barrier.wait()
    t0 = time.perf_counter_ns()
    kernel(buf, slot, iters, seed)
    times[slot_key(slot)] = time.perf_counter_ns() - t0


def slot_key(slot):
    return slot


def _run_threads(kernel, stride, ctx, iters):
    buf = make_aligned_buffer(NUM_THREADS * max(stride, 1) + 16)
    barrier = threading.Barrier(NUM_THREADS + 1)
    per_thread = {}
    threads = []
    base_seed = ctx["instance_seed"]
    for t in range(NUM_THREADS):
        core = ctx["core_ids"][t] if ctx["affinity_supported"] else None
        slot = t * stride
        seed = (base_seed + t) & 0xFFFFFFFFFFFFFFFF
        th = threading.Thread(
            target=_thread_target,
            args=(kernel, buf, slot, iters, seed, core, barrier, per_thread),
        )
        threads.append(th)
        th.start()

    ru0 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    barrier.wait()
    t0 = time.perf_counter_ns()
    for th in threads:
        th.join()
    elapsed = time.perf_counter_ns() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None

    vol = (ru1.ru_nvcsw - ru0.ru_nvcsw) if resource else None
    invol = (ru1.ru_nivcsw - ru0.ru_nivcsw) if resource else None
    finals = [int(buf[t * stride]) for t in range(NUM_THREADS)]
    return elapsed, vol, invol, finals


# ==============================================================================
# TASK 3: Multi-threaded stride sweep
# ==============================================================================
def run_stride_benchmark(ctx: dict, iters: int = ITERATIONS_PER_THREAD) -> dict:
    # Warm-up / JIT compile on tiny input (not timed)
    warm = make_aligned_buffer(32)
    worker_strided_accumulate(warm, 0, 10, 1)
    worker_private_accumulate(warm, 0, 10, 1)

    results = {}
    for stride in STRIDES:
        elapsed, vol, invol, finals = _run_threads(worker_strided_accumulate, stride, ctx, iters)
        results[f"stride_{stride}"] = {
            "stride_elements": stride,
            "stride_bytes": stride * 8,
            "elapsed_ns": elapsed,
            "ns_per_update_per_thread": elapsed / iters,
            "ns_per_update_aggregate": elapsed / (iters * NUM_THREADS),
            "voluntary_ctx_switches": vol,
            "involuntary_ctx_switches": invol,
            "final_values": finals,
        }
        print(f"[stride={stride:2d}] {elapsed/1e6:10.2f} ms  ({elapsed/iters:.3f} ns/update/thread)")

    # Task 4 baseline
    elapsed, vol, invol, finals = _run_threads(worker_private_accumulate, 8, ctx, iters)
    results["private_baseline"] = {
        "stride_elements": 8,
        "stride_bytes": 64,
        "elapsed_ns": elapsed,
        "ns_per_update_per_thread": elapsed / iters,
        "ns_per_update_aggregate": elapsed / (iters * NUM_THREADS),
        "voluntary_ctx_switches": vol,
        "involuntary_ctx_switches": invol,
        "final_values": finals,
    }
    print(f"[private ] {elapsed/1e6:10.2f} ms  ({elapsed/iters:.3f} ns/update/thread)")

    s1 = results["stride_1"]["elapsed_ns"]
    s8 = results["stride_8"]["elapsed_ns"]
    results["slowdown_stride1_vs_stride8"] = s1 / s8
    results["slowdown_stride1_vs_private"] = s1 / results["private_baseline"]["elapsed_ns"]
    results["slowdown_stride8_vs_private"] = s8 / results["private_baseline"]["elapsed_ns"]
    print(f"Slowdown stride1/stride8 = {s1 / s8:.2f}x")
    return results


# ==============================================================================
# TASK 5: Parity + LLVM IR audit + manifest
# ==============================================================================
def audit_llvm_and_parity(ctx: dict, results: dict) -> dict:
    # --- Bit-exact parity: every layout and the privatized baseline must agree ---
    reference = results["stride_8"]["final_values"]
    parity = True
    for key in ("stride_1", "stride_4", "stride_8", "stride_16", "private_baseline"):
        if results[key]["final_values"] != reference:
            parity = False
            print(f"PARITY FAIL: {key}: {results[key]['final_values']} != {reference}")

    # --- LLVM IR inspection ---
    sig = worker_strided_accumulate.signatures[0]
    ir = worker_strided_accumulate.inspect_llvm(sig)
    has_i64_store = "store i64" in ir
    has_i64_load = "load i64" in ir
    # Python object fallback would reference NRT/PyObject API calls
    object_fallback = any(tok in ir for tok in ("PyObject_", "PyLong_", "PyNumber_", "Py_DecRef"))
    ir_lines = [l.strip() for l in ir.splitlines() if "store i64" in l or "load i64" in l]

    verification_passed = bool(parity and has_i64_store and has_i64_load and not object_fallback)

    manifest = {
        "git_username": ctx["git_user"],
        "instance_seed": ctx["instance_seed"],
        "cpu_model": ctx["cpu_model"],
        "cpu_signature": ctx["cpu_sig"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size_bytes": ctx["cache_line_size"],
        "affinity_supported": ctx["affinity_supported"],
        "core_ids": ctx["core_ids"],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "python_version": ctx["python"],
        "platform": ctx["platform"],
        "timings": results,
        "llvm": {
            "has_store_i64": has_i64_store,
            "has_load_i64": has_i64_load,
            "python_object_fallback": object_fallback,
            "sample_memory_ops": ir_lines[:12],
        },
        "bit_exact_parity": parity,
        "verification_passed": verification_passed,
    }
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"verification_passed = {verification_passed}  (manifest written)")
    return manifest


def main():
    ctx = init_system_context(GIT_USERNAME)
    print(f"Cores={ctx['logical_cores']} line={ctx['cache_line_size']}B "
          f"seed={ctx['instance_seed']:#018x} affinity={ctx['affinity_supported']}")
    results = run_stride_benchmark(ctx)
    audit_llvm_and_parity(ctx, results)


if __name__ == "__main__":
    main()
