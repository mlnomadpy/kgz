"""
Health monitoring for Kaggle kernels — GPU/TPU usage, training progress parsing.
"""

import re
import time


def parse_training_progress(log_line):
    """
    Parse training metrics from a log line.
    Handles common formats from JAX, PyTorch, flaxchat, nanochat.

    Returns dict with parsed fields.
    """
    metrics = {}

    m = re.search(r'step\s*[:=]?\s*(\d+)', log_line, re.IGNORECASE)
    if m: metrics['step'] = int(m.group(1))

    m = re.search(r'step\s*\d+\s*/\s*(\d+)', log_line, re.IGNORECASE)
    if m: metrics['total_steps'] = int(m.group(1))

    m = re.search(r'loss\s*[:=]?\s*([\d.]+)', log_line, re.IGNORECASE)
    if m: metrics['loss'] = float(m.group(1))

    m = re.search(r'lr\s*[:=]?\s*([\d.e-]+)', log_line, re.IGNORECASE)
    if m:
        try: metrics['lr'] = float(m.group(1))
        except ValueError: pass

    m = re.search(r'tok/?s\s*[:=]?\s*([\d,]+)', log_line, re.IGNORECASE)
    if m: metrics['tok_per_sec'] = int(m.group(1).replace(',', ''))

    m = re.search(r'(\d+\.?\d*)%', log_line)
    if m: metrics['percent'] = float(m.group(1))

    m = re.search(r'mfu\s*[:=]?\s*([\d.]+)', log_line, re.IGNORECASE)
    if m: metrics['mfu'] = float(m.group(1))

    return metrics


def estimate_eta(metrics):
    """Estimate seconds remaining from parsed metrics."""
    if 'step' in metrics and 'total_steps' in metrics and 'dt' in metrics:
        remaining = metrics['total_steps'] - metrics['step']
        return remaining * metrics['dt']
    return None


class KernelMonitor:
    """
    Monitor a running Kaggle kernel for health and training progress.

    Usage:
        mon = KernelMonitor(kernel)
        status = mon.check()
        mon.check_pretty()
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self._last_output = ""

    def check(self):
        """Full health check. Returns status dict."""
        status = {"timestamp": time.time()}

        # Kernel status
        try:
            status["kernel_status"] = self.kernel.status()
        except Exception:
            status["kernel_status"] = "unreachable"

        # Resources (GPU/TPU, CPU, RAM)
        try:
            res = self.kernel.resources()
            status.update(res)
        except Exception:
            pass

        # Parse latest output for training metrics
        if self.kernel._history:
            last = self.kernel._history[-1]
            output = last.get("output", "")
            if output:
                for line in output.strip().split("\n"):
                    m = parse_training_progress(line)
                    if m:
                        status["training"] = m
                        eta = estimate_eta(m)
                        if eta:
                            status["eta_seconds"] = round(eta)

        return status

    def check_pretty(self):
        """Print formatted health dashboard."""
        s = self.check()
        G, R, Y, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

        print(f"\n  Kaggle Kernel Health")
        print(f"  {'='*50}")

        # Status
        ks = s.get("kernel_status", "?")
        c = G if ks == "idle" or ks == "busy" else R
        print(f"  Kernel:  {c}{ks}{RESET}")

        # Backend
        backend = s.get("backend", "?")
        devices = s.get("device_count", "?")
        print(f"  Backend: {backend} ({devices} devices)")

        # GPU/TPU
        if "gpus" in s:
            for i, g in enumerate(s["gpus"]):
                util = g.get("utilization", 0)
                mem = g.get("memory_used_mb", 0)
                total = g.get("memory_total_mb", 0)
                c = G if util > 10 else Y
                print(f"  GPU {i}:   {c}{util}% util{RESET}, {mem}/{total} MB")

        # CPU/RAM
        cpu = s.get("cpu_percent", "?")
        ram_used = s.get("ram_used_gb", "?")
        ram_total = s.get("ram_total_gb", "?")
        print(f"  CPU:     {cpu}%")
        print(f"  RAM:     {ram_used}/{ram_total} GB")

        # Training
        t = s.get("training", {})
        if t:
            parts = []
            if "step" in t:
                step_s = f"step {t['step']}"
                if "total_steps" in t: step_s += f"/{t['total_steps']}"
                parts.append(step_s)
            if "loss" in t: parts.append(f"loss {t['loss']:.4f}")
            if "tok_per_sec" in t: parts.append(f"{t['tok_per_sec']:,} tok/s")
            if parts:
                print(f"  Train:   {' | '.join(parts)}")
            if "eta_seconds" in s:
                eta = s["eta_seconds"]
                eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"
                print(f"  ETA:     ~{eta_str}")

        # Quota
        try:
            qt, dt = self.kernel.quota()
            remaining = qt.remaining(dt)
            session_left = qt.session_time_left(dt)
            print(f"  Quota:   {remaining:.1f}h remaining ({dt.upper()})")
            print(f"  Session: {session_left:.1f}h before expiry")
        except Exception:
            pass

        print()
        return s
