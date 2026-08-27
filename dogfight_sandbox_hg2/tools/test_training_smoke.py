# -*- coding: utf-8 -*-
"""Integration smoke test: boots the real sandbox (1v1 mission) and runs a few
hundred training steps of each algorithm through training.train, verifying the
observe -> decide -> step -> update loop and checkpoint artifacts.

Usage:  python tools/test_training_smoke.py [--keep]
(uses the system Python with torch; sandbox uses the bundled interpreter)
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ALGO_SETTINGS = {
    "ppo": ["--set", "rollout_size=256", "batch_size=64", "epochs=2"],
    "sac": ["--set", "learning_starts=100", "batch_size=64"],
    "rainbow": ["--set", "learning_starts=100", "batch_size=32",
                "n_step=3", "v_min=-150", "v_max=150"],
}

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print("%-42s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


def wait_port(host, port, timeout_s=90):
    import socket
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=1.0)
            s.close()
            return True
        except OSError:
            time.sleep(1.0)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--host", default="192.168.1.103")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    run_root = os.path.join(ROOT, "checkpoints", "_smoke")
    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", run_root], capture_output=True)

    sandbox = subprocess.Popen(
        [os.path.join(ROOT, "dogfight_sandbox_hg2", "bin", "python", "python.exe"),
         "main.py", "auto_network", "mission=1"],
        cwd=os.path.join(ROOT, "dogfight_sandbox_hg2", "source"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        check("sandbox boots (1v1 mission)", wait_port(args.host, 50888))

        for algo, extra in ALGO_SETTINGS.items():
            cmd = [sys.executable, "-u", "-m", "training.train",
                   "--algo", algo, "--env", "oneVSone", "--host", args.host,
                   "--timesteps", str(args.steps), "--device", "cpu",
                   "--save-dir", run_root, "--checkpoint-interval", "1000",
                   "--name", algo] + extra
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
            tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
            ok = proc.returncode == 0
            check("%s: %d steps run" % (algo, args.steps), ok,
                  tail[0][:70] if ok else (proc.stderr or "").strip().splitlines()[-1][:120])

            model = os.path.join(run_root, algo, "model_final.pt")
            extra_pt = os.path.join(run_root, algo, "extra_final.pt")
            check("%s: checkpoint written" % algo,
                  os.path.exists(model) and os.path.exists(extra_pt))

            if ok and os.path.exists(model):
                from training.algorithms import load_algo
                loaded = load_algo(model, device="cpu")
                check("%s: checkpoint loadable" % algo, loaded.name == algo)
    finally:
        if not args.keep:
            sandbox.terminate()
            try:
                sandbox.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sandbox.kill()

    failed = [n for n, ok in results if not ok]
    print("\n%d/%d checks passed" % (len(results) - len(failed), len(results)))
    if failed:
        print("FAILED:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
