"""Benchmark mjx.step speed for MyoFullBody.

Usage:
    uv run python scripts/benchmark_step.py
    uv run python scripts/benchmark_step.py --impl jax
    uv run python scripts/benchmark_step.py --impl cpu
    uv run python scripts/benchmark_step.py --num-envs 8192
    uv run python scripts/benchmark_step.py --impl transfer --num-envs 128 --nstep 20 --obs-dim 1024
"""

import argparse
import os
import time

import mujoco
import numpy as np
from musclemimic_models import get_xml_path

# Config from conf_fullbody.yaml + MjxMyoFullBody
MODEL_PATH = get_xml_path("myofullbody").as_posix()
N_SUBSTEPS = 5
TIMESTEP = 0.002
SOLVER_ITERATIONS = 4
SOLVER_LS_ITERATIONS = 8
NCONMAX = 96
NJMAX = 768


def _print_system_info():
    """Print system resource info for reproducibility."""
    import subprocess

    print("=" * 60)
    print("System Info:")

    # CPU
    cpu_count = os.cpu_count()
    try:
        affinity = os.sched_getaffinity(0)
        usable_cpus = len(affinity)
    except AttributeError:
        usable_cpus = cpu_count
    print(f"  CPUs (total/usable):  {cpu_count} / {usable_cpus}")

    # CPU model
    try:
        model = subprocess.check_output("lscpu | grep 'Model name'", shell=True, text=True).strip().split(":")[-1].strip()
        print(f"  CPU model:            {model}")
    except Exception:
        pass

    # Memory
    try:
        mem = subprocess.check_output("free -h | grep Mem", shell=True, text=True).strip().split()
        print(f"  RAM (total/avail):    {mem[1]} / {mem[6]}")
    except Exception:
        pass

    # cgroup CPU limits (RunAI/K8s)
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().strip().split()
            if quota != "max":
                effective_cpus = int(quota) / int(period)
                print(f"  cgroup CPU limit:     {effective_cpus:.1f} cores")
            else:
                print(f"  cgroup CPU limit:     unlimited")
    except FileNotFoundError:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                quota = int(f.read().strip())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read().strip())
            if quota > 0:
                print(f"  cgroup CPU limit:     {quota / period:.1f} cores")
            else:
                print(f"  cgroup CPU limit:     unlimited")
        except Exception:
            pass

    # cgroup memory limits
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            mem_limit = f.read().strip()
            if mem_limit != "max":
                print(f"  cgroup mem limit:     {int(mem_limit) / 1024**3:.1f} GB")
            else:
                print(f"  cgroup mem limit:     unlimited")
    except Exception:
        pass

    print("=" * 60)


def _get_effective_cpus() -> int:
    """Get actual usable CPU count, respecting cgroup limits (RunAI/K8s)."""
    fallback = os.cpu_count() or 1
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().strip().split()
            if quota != "max":
                return max(1, int(int(quota) / int(period)))
    except (FileNotFoundError, ValueError):
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        if quota > 0:
            return max(1, int(quota / period))
    except (FileNotFoundError, ValueError):
        pass
    return fallback


def _load_model(impl: str) -> mujoco.MjModel:
    """Load model via MjSpec with correct settings."""
    spec = mujoco.MjSpec.from_file(MODEL_PATH)
    if impl == "warp":
        spec.nconmax = NCONMAX
        spec.njmax = NJMAX
    m = spec.compile()
    m.opt.timestep = TIMESTEP
    m.opt.iterations = SOLVER_ITERATIONS
    m.opt.ls_iterations = SOLVER_LS_ITERATIONS
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    return m


def _get_rss_mb() -> float:
    """Get current process RSS (resident memory) in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except Exception:
        pass
    return 0.0


def _print_results(impl: str, num_envs: int, nstep: int, times: list[float]):
    avg = sum(times) / len(times)
    total_steps = nstep * num_envs
    sps = total_steps / avg
    print(f"\n{'='*60}")
    print(f"  Backend:           {impl}")
    print(f"  Envs:              {num_envs}")
    print(f"  Total SPS:         {sps:,.0f}")
    print(f"  Per-env SPS:       {sps / num_envs:,.0f}")
    print(f"  Realtime factor:   {total_steps * TIMESTEP / avg:.1f}x")
    print(f"  Time/step:         {1e6 * avg / total_steps:.2f} µs")
    print(f"  Ctrl SPS (x{N_SUBSTEPS}):    {sps / N_SUBSTEPS:,.0f}")
    print(f"{'='*60}")


def benchmark_gpu(impl: str = "warp", num_envs: int = 128, nstep: int = 1000, unroll: int = 4):
    """Benchmark MJX (warp/jax) backend."""
    import jax
    from mujoco import mjx

    naconmax = NCONMAX * num_envs

    m = _load_model(impl)
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}, nbody={m.nbody}, ngeom={m.ngeom}")
    print(f"Backend: {impl}, envs: {num_envs}, nstep: {nstep}")

    mx = mjx.put_model(m, impl=impl)

    @jax.jit
    @jax.vmap
    def init(key):
        if impl == "warp":
            d = mjx.make_data(m, impl=impl, nconmax=NCONMAX, njmax=NJMAX, naconmax=naconmax)
        else:
            d = mjx.make_data(m, impl=impl)
        qvel = 0.01 * jax.random.normal(key, shape=(m.nv,))
        return d.replace(qvel=qvel)

    print(f"Initializing {num_envs} environments...")
    keys = jax.random.split(jax.random.key(0), num_envs)
    d = init(keys)
    jax.block_until_ready(d)

    @jax.jit
    @jax.vmap
    def run_steps(d):
        def step_fn(d, _):
            return mjx.step(mx, d), None
        return jax.lax.scan(step_fn, d, None, length=nstep, unroll=unroll)[0]

    print("JIT compiling...")
    t0 = time.perf_counter()
    d_out = run_steps(d)
    jax.block_until_ready(d_out)
    print(f"JIT time: {time.perf_counter() - t0:.2f}s")

    total_steps = nstep * num_envs
    print(f"\nBenchmarking {total_steps:,} total steps...")
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        d_out = run_steps(d)
        jax.block_until_ready(d_out)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i}: {elapsed:.3f}s | {total_steps / elapsed:,.0f} SPS")

    _print_results(impl, num_envs, nstep, times)


def benchmark_cpu(num_envs: int = 128, nstep: int = 1000, nthread: int = 0):
    """Benchmark CPU MuJoCo with mujoco.rollout (multi-threaded)."""
    from mujoco.rollout import rollout

    m = _load_model("cpu")
    if nthread <= 0:
        nthread = min(num_envs, _get_effective_cpus())

    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}, nbody={m.nbody}, ngeom={m.ngeom}")
    print(f"Backend: cpu (mujoco.rollout), envs: {num_envs}, nthread: {nthread}, nstep: {nstep}")

    rss_before = _get_rss_mb()

    # Create per-thread MjData (rollout uses len(data) as thread count)
    data = [mujoco.MjData(m) for _ in range(nthread)]

    # Initial states: small random qvel to avoid static equilibrium
    # rollout expects state size = mj_stateSize(m, mjSTATE_FULLPHYSICS)
    rng = np.random.default_rng(0)
    nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
    initial_state = np.zeros((num_envs, nstate))
    # FULLPHYSICS state layout: [time(1), qpos(nq), qvel(nv), act(na)]
    initial_state[:, 1 + m.nq:1 + m.nq + m.nv] = 0.01 * rng.standard_normal((num_envs, m.nv))

    # Small random control noise
    control = 0.01 * rng.standard_normal((num_envs, nstep, m.nu)).astype(np.float64)

    # Memory breakdown
    rss_after = _get_rss_mb()
    print(f"\nMemory:")
    print(f"  MjData x{nthread}:         ~{nthread * initial_state[:1].nbytes / 1024 / 1024:.0f} MB (estimate)")
    print(f"  Control array:         {control.nbytes / 1024 / 1024:.0f} MB")
    print(f"  State array:           {num_envs * nstep * nstate * 8 / 1024 / 1024:.0f} MB (output)")
    print(f"  Process RSS:           {rss_after:.0f} MB (delta: +{rss_after - rss_before:.0f} MB)")

    # Warmup
    print("\nWarming up...")
    _ = rollout(m, data, initial_state, control, nstep=nstep)

    total_steps = nstep * num_envs
    print(f"\nBenchmarking {total_steps:,} total steps...")
    times = []
    for i in range(3):
        # Measure both wall time and CPU time to detect throttling
        cpu_t0 = time.process_time()
        wall_t0 = time.perf_counter()
        _ = rollout(m, data, initial_state, control, nstep=nstep)
        wall_elapsed = time.perf_counter() - wall_t0
        cpu_elapsed = time.process_time() - cpu_t0
        times.append(wall_elapsed)
        # cpu_time/wall_time > 1 means multiple cores used; < nthread means throttled
        cpu_ratio = cpu_elapsed / wall_elapsed if wall_elapsed > 0 else 0
        print(f"  Run {i}: {wall_elapsed:.3f}s | {total_steps / wall_elapsed:,.0f} SPS | "
              f"CPU/wall: {cpu_ratio:.1f}x (ideal: {nthread}x)")

    rss_peak = _get_rss_mb()
    print(f"\n  Peak RSS: {rss_peak:.0f} MB")
    _print_results("cpu", num_envs, nstep, times)


def _measure_cpu_sim_sps(num_envs: int, nstep: int, nthread: int) -> tuple[float, int, int]:
    """Measure CPU simulator SPS for transfer-overhead estimation."""
    from mujoco.rollout import rollout

    m = _load_model("cpu")
    if nthread <= 0:
        nthread = min(num_envs, _get_effective_cpus())

    # Keep control tensor bounded to avoid OOM for large env counts.
    # control bytes = num_envs * nstep * nu * 8 (float64)
    per_step_bytes = num_envs * m.nu * np.dtype(np.float64).itemsize
    max_nstep = max(1, int((1024**3) // max(1, per_step_bytes)))  # cap control at ~1GB
    if nstep > max_nstep:
        print(f"  [transfer] sim_nstep reduced from {nstep} to {max_nstep} to keep control buffer <=1GB")
        nstep = max_nstep

    data = [mujoco.MjData(m) for _ in range(nthread)]
    rng = np.random.default_rng(0)
    nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
    initial_state = np.zeros((num_envs, nstate))
    initial_state[:, 1 + m.nq:1 + m.nq + m.nv] = 0.01 * rng.standard_normal((num_envs, m.nv))
    control = 0.01 * rng.standard_normal((num_envs, nstep, m.nu)).astype(np.float64)

    _ = rollout(m, data, initial_state, control, nstep=nstep)  # warmup

    total_steps = nstep * num_envs
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = rollout(m, data, initial_state, control, nstep=nstep)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    sps = total_steps / avg
    return sps, nthread, nstep


def benchmark_transfer(
    num_envs: int = 128,
    num_steps: int = 20,
    obs_dim: int = 1024,
    nthread: int = 0,
    sim_nstep: int = 200,
    n_trials: int = 200,
):
    """Benchmark transfer and estimate simulation SPS after transfer overhead."""
    import jax

    try:
        gpu_devices = jax.devices("gpu")
    except Exception:
        gpu_devices = []
    if not gpu_devices:
        print("\nNo GPU detected. Transfer benchmark requires a GPU.")
        return
    device = gpu_devices[0]

    m = _load_model("warp")
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}")
    print(f"Envs: {num_envs}, Device: {device}")
    print(f"PPO rollout: num_steps={num_steps}, obs_dim={obs_dim}")

    # Measure CPU sim baseline so overhead-adjusted SPS can be computed directly.
    print("\nMeasuring CPU simulator baseline SPS...")
    sim_sps, used_threads, used_sim_nstep = _measure_cpu_sim_sps(num_envs=num_envs, nstep=sim_nstep, nthread=nthread)
    sim_step_ms = 1e3 * num_envs / sim_sps
    print(f"  CPU sim baseline: {sim_sps:,.0f} SPS (nthread={used_threads}, sim_nstep={used_sim_nstep})")
    print(f"  CPU step latency (for {num_envs} envs): {sim_step_ms:.3f} ms")

    # Realistic array sizes from training pipeline
    test_cases = [
        ("qpos",         (num_envs, m.nq)),
        ("qvel",         (num_envs, m.nv)),
        ("action",       (num_envs, m.nu)),
        ("obs",          (num_envs, obs_dim)),
        ("state (full)", (num_envs, m.nq + m.nv + m.nu)),
    ]

    max_staging_mb = 256.0

    def _block_tree(tree):
        for leaf in jax.tree_util.tree_leaves(tree):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()

    def _iter_fresh_device_batches(cpu_arr: np.ndarray, total: int):
        """Yield fresh device arrays in bounded batches for realistic D2H timing."""
        bytes_per_arr = max(1, int(cpu_arr.nbytes))
        max_batch = max(1, int((max_staging_mb * 1024**2) // bytes_per_arr))
        remaining = total
        while remaining > 0:
            batch_size = min(max_batch, remaining)
            batch = [jax.device_put(cpu_arr, device=device) for _ in range(batch_size)]
            for arr in batch:
                arr.block_until_ready()
            yield batch
            remaining -= batch_size

    print(f"\n  {'Array':<18} {'Shape':<18} {'Size':>8}  {'H2D':>9}  {'D2H':>9}  {'H2D BW':>10}  {'D2H BW':>10}")
    print("  " + "-" * 88)

    for name, shape in test_cases:
        cpu_arr = np.random.randn(*shape).astype(np.float32)
        size_mb = cpu_arr.nbytes / 1024**2

        # Warmup
        gpu_arr = jax.device_put(cpu_arr, device=device)
        gpu_arr.block_until_ready()
        _ = jax.device_get(gpu_arr)

        # H2D: CPU numpy -> GPU
        t0 = time.perf_counter()
        for _ in range(n_trials):
            gpu_arr = jax.device_put(cpu_arr, device=device)
            gpu_arr.block_until_ready()
        h2d = (time.perf_counter() - t0) / n_trials

        # D2H: GPU -> CPU numpy using fresh buffers (avoid cache-hit optimistic timing).
        d2h_total = 0.0
        d2h_count = 0
        for gpu_batch in _iter_fresh_device_batches(cpu_arr, n_trials):
            t0 = time.perf_counter()
            for arr in gpu_batch:
                _ = jax.device_get(arr)
            d2h_total += time.perf_counter() - t0
            d2h_count += len(gpu_batch)
        d2h = d2h_total / d2h_count

        h2d_bw = size_mb / h2d / 1024  # GB/s
        d2h_bw = size_mb / d2h / 1024  # GB/s

        print(f"  {name:<18} {str(shape):<18} {size_mb:>6.2f}MB  "
              f"{h2d*1000:>7.3f}ms  {d2h*1000:>7.3f}ms  "
              f"{h2d_bw:>7.2f}GB/s  {d2h_bw:>7.2f}GB/s")

    # Round-trip: obs H2D + action D2H (models CPU sim + GPU policy scenario)
    print(f"\n  Round-trip (obs H2D + action D2H):")
    obs_cpu = np.random.randn(num_envs, obs_dim).astype(np.float32)

    # Warmup
    obs_gpu = jax.device_put(obs_cpu, device=device)
    obs_gpu.block_until_ready()
    action_gpu = obs_gpu[:, :m.nu]
    _ = jax.device_get(action_gpu)

    rt_total = 0.0
    for _ in range(n_trials):
        t0 = time.perf_counter()
        obs_gpu = jax.device_put(obs_cpu, device=device)
        obs_gpu.block_until_ready()
        # Cheap GPU-side action extraction to force realistic D2H size (num_envs, nu).
        action_gpu = obs_gpu[:, :m.nu]
        _ = jax.device_get(action_gpu)
        rt_total += time.perf_counter() - t0

    rt = rt_total / n_trials
    action_bytes = num_envs * m.nu * np.dtype(np.float32).itemsize
    total_mb = (obs_cpu.nbytes + action_bytes) / 1024**2
    print(f"    Latency: {rt*1000:.3f}ms | Total data: {total_mb:.2f}MB | "
          f"Effective BW: {total_mb / rt / 1024:.2f}GB/s")

    # PPO rollout upload CPU->GPU once/update
    rollout_cpu = {
        "obs": np.random.randn(num_steps, num_envs, obs_dim).astype(np.float32),
        "action": np.random.randn(num_steps, num_envs, m.nu).astype(np.float32),
        "reward": np.random.randn(num_steps, num_envs).astype(np.float32),
        "done": np.random.randint(0, 2, size=(num_steps, num_envs), dtype=np.bool_),
        "absorbing": np.random.randint(0, 2, size=(num_steps, num_envs), dtype=np.bool_),
        "value": np.random.randn(num_steps, num_envs).astype(np.float32),
        "log_prob": np.random.randn(num_steps, num_envs).astype(np.float32),
        "last_obs": np.random.randn(num_envs, obs_dim).astype(np.float32),
        "last_value": np.random.randn(num_envs).astype(np.float32),
    }
    rollout_mb = sum(arr.nbytes for arr in rollout_cpu.values()) / 1024**2

    # Warmup
    rollout_gpu = jax.device_put(rollout_cpu, device=device)
    _block_tree(rollout_gpu)

    t0 = time.perf_counter()
    for _ in range(n_trials):
        rollout_gpu = jax.device_put(rollout_cpu, device=device)
        _block_tree(rollout_gpu)
    upload = (time.perf_counter() - t0) / n_trials
    print(f"\n  PPO rollout upload (CPU -> GPU once/update):")
    print(f"    Data/update: {rollout_mb:.2f}MB | Latency: {upload*1000:.3f}ms | BW: {rollout_mb / upload / 1024:.2f}GB/s")

    # Convert transfer overhead into effective simulation throughput.
    steps_per_update = num_steps * num_envs
    sim_step_s = num_envs / sim_sps
    step_transfer_s = rt
    upload_per_step_s = upload / max(1, num_steps)
    eff_step_time_no_upload = sim_step_s + step_transfer_s
    eff_step_time_with_upload = sim_step_s + step_transfer_s + upload_per_step_s
    eff_sps_no_upload = num_envs / eff_step_time_no_upload
    eff_sps_with_upload = num_envs / eff_step_time_with_upload
    eff_update_sps_with_upload = 1.0 / (num_steps * (sim_step_s + step_transfer_s) + upload)
    eff_env_steps_s_with_upload = steps_per_update * eff_update_sps_with_upload

    print(f"\n  Simulation SPS after transfer overhead:")
    print(f"    Baseline (CPU sim only):      {sim_sps:,.0f} SPS")
    print(f"    + step transfer only:         {eff_sps_no_upload:,.0f} SPS")
    print(f"    + step transfer + upload avg: {eff_sps_with_upload:,.0f} SPS")
    print(f"    Slowdown vs baseline:         {sim_sps / eff_sps_with_upload:.2f}x")
    print(f"    Update throughput (with upload): {eff_update_sps_with_upload:,.2f} updates/s "
          f"| {eff_env_steps_s_with_upload:,.0f} env-steps/s")

    print(f"\n{'='*60}")
    print(f"  Note: This models CPU sim + GPU policy + GPU PPO training upload.")
    print(f"  For pure MJX training, per-step host/device transfer is near zero.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", default="warp", choices=["jax", "warp", "cpu", "transfer"])
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument(
        "--nstep",
        type=int,
        default=None,
        help="Steps per run (GPU/CPU) or PPO rollout num_steps (transfer); defaults: 200 GPU, 8000 CPU, 20 transfer",
    )
    parser.add_argument("--obs-dim", type=int, default=1024, help="Observation dimension for transfer benchmark")
    parser.add_argument("--transfer-trials", type=int, default=200, help="Number of trials for transfer timing")
    parser.add_argument("--sim-nstep", type=int, default=200, help="CPU sim steps used to estimate baseline SPS in transfer mode")
    parser.add_argument("--unroll", type=int, default=1)
    parser.add_argument("--nthread", type=int, default=64, help="CPU threads (0=auto)")
    args = parser.parse_args()

    _print_system_info()
    if args.impl == "cpu":
        nstep = args.nstep or 8_000
        benchmark_cpu(num_envs=args.num_envs, nstep=nstep, nthread=args.nthread)
    elif args.impl == "transfer":
        nstep = args.nstep or 20
        benchmark_transfer(
            num_envs=args.num_envs,
            num_steps=nstep,
            obs_dim=args.obs_dim,
            nthread=args.nthread,
            sim_nstep=args.sim_nstep,
            n_trials=args.transfer_trials,
        )
    else:
        nstep = args.nstep or 200
        benchmark_gpu(impl=args.impl, num_envs=args.num_envs, nstep=nstep, unroll=args.unroll)
