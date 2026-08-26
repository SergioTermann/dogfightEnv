# -*- coding: utf-8 -*-
"""Mission commander: observes the dogfight sandbox over TCP/IP and assigns a task
(engage / patrol / retreat / hold) to every plane of one side.

The decision engine is pluggable (config "engine"):
  - "rule": deterministic tactician, no external dependency (default demo)
  - "llm": OpenAI-compatible chat/completions endpoint (set llm.api_key etc.)

The commander never drives the simulation clock: the sandbox free-runs at 60 fps
and this process only polls states and pushes commands, so decision latency
(rule instant, LLM seconds) cannot stall the battle.

Usage (works with the sandbox's embedded python or any system python >= 3.7):
  python commander.py [--config path/to/config.json] [--dry-run] [--once] [--duration N]

Run alongside a sandbox started with e.g.:
  cd dogfight_sandbox_hg2/source && ../bin/python/python.exe main.py auto_network mission=2
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)  # embedded python (._pth) does not add the script dir
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'dogfight_sandbox_hg2', 'source'))

import tactician  # noqa: E402  (local module next to this file)

from dogfight_sandbox_hg2.network_client_example import dogfight_client as df  # noqa: E402

# In-view labels stay ASCII: the sandbox fonts (default.ttf/Furore.otf) have no
# CJK glyphs, Chinese would render as blanks. Console/log stay Chinese.
TASK_LABELS = {
    "engage": u"ENGAGE",
    "patrol": u"PATROL",
    "retreat": u"RETREAT",
    "hold": u"HOLD",
}
TASK_LABEL_DEAD = u"DOWN"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def make_engine(cfg):
    if cfg.get("engine") == "llm":
        llm = cfg.get("llm", {})
        engine = tactician.LLMTactician(
            api_base=llm.get("api_base", ""),
            api_key=llm.get("api_key", ""),
            model=llm.get("model", "glm-4-flash"),
            temperature=float(llm.get("temperature", 0.2)),
            timeout_s=float(llm.get("timeout_s", 25)),
        )
        if not engine.api_key:
            print("[commander] engine=llm but llm.api_key is empty -> falling back to rule engine")
            return tactician.RuleTactician()
        return engine
    return tactician.RuleTactician()


class Commander:
    def __init__(self, cfg, dry_run=False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.engine = make_engine(cfg)
        self.side = cfg.get("side", "ennemies")
        self.period_s = float(cfg.get("decision_period_s", 10))
        self.poll_s = float(cfg.get("poll_interval_s", 0.5))
        self.missiles_poll_s = float(cfg.get("missiles_poll_s", 5))
        self.host = cfg.get("host", "192.168.1.103")
        self.port = int(cfg.get("port", 50888))

        self.planes = []
        self.own = []
        self.applied = {}          # plane -> last applied assignment
        self.missiles = {}         # plane -> remaining missiles
        self.missiles_t = 0.0
        self.last_decision_t = -1e9
        self.last_alive_sig = None
        self.sim_t0 = None
        self.log_path = os.path.join(HERE, "decisions.jsonl")

    # ------------------------------------------------------------------ setup

    def connect(self, timeout_s=60):
        import socket
        deadline = time.time() + timeout_s
        # df.connect retries forever internally, so only call it once the port
        # answers; otherwise a dead sandbox hangs the commander.
        while time.time() < deadline:
            try:
                probe = socket.create_connection((self.host, self.port), timeout=1.0)
                probe.close()
            except OSError:
                time.sleep(1.0)
                continue
            try:
                df.connect(self.host, self.port)
                self.planes = df.get_planes_list()
                if self.planes:
                    break
            except Exception as exc:
                print("[commander] connect attempt failed: %r" % (exc,))
                time.sleep(1.0)
        else:
            raise RuntimeError("cannot connect to sandbox at %s:%d" % (self.host, self.port))
        print("[commander] connected: %d planes %s" % (len(self.planes), self.planes))

        states = self.poll_states()
        self.own = [n for n in self.planes
                    if (states.get(n, {}).get("nationality") == 1) == (self.side == "allies")]
        print("[commander] engine=%s side=%s planes=%s" % (self.engine.name, self.side, self.own))

        # sparring partner: activate the built-in IA on the opposite side
        sparring = ((self.side == "ennemies" and self.cfg.get("blue_ia", False))
                    or (self.side == "allies" and self.cfg.get("red_ia", False)))
        if sparring:
            for n in self.planes:
                if n not in self.own:
                    df.activate_IA(n)
                    print("[commander] sparring IA activated on %s" % n)

    # ------------------------------------------------------------------ polls

    def poll_states(self):
        states = {}
        for name in self.planes:
            try:
                states[name] = df.get_plane_state(name)
            except Exception as exc:
                print("[commander] state poll failed for %s: %r" % (name, exc))
        if self.sim_t0 is None:
            self.sim_t0 = states[self.planes[0]].get("timestamp", 0.0) if states else 0.0
        return states

    def poll_missiles(self, now):
        if now - self.missiles_t < self.missiles_poll_s:
            return
        self.missiles_t = now
        for name in self.planes:
            try:
                slots = df.get_machine_missiles_list(name)
                self.missiles[name] = sum(1 for s in slots if s)
            except Exception:
                pass

    # ------------------------------------------------------------------ decisions

    def decide(self, states):
        any_state = next(iter(states.values()), {})
        sim_time = any_state.get("timestamp", 0.0) - (self.sim_t0 or 0.0)
        situation = tactician.build_situation(states, self.missiles, self.side, sim_time)
        last = [self.applied.get(n) for n in self.own if n in self.applied] or None
        result = self.engine.decide(situation, last)
        if result is None:
            print("[commander] engine returned no decision, keeping previous assignments")
            return
        self.apply(result["assignments"], states)
        self.log_decision(sim_time, result)

    def apply(self, assignments, states):
        live_own = {a["plane"] for a in assignments}
        for a in assignments:
            name = a["plane"]
            if not tactician.plane_alive(states.get(name)):
                if self.applied.get(name, {}).get("task") != "_dead":
                    self.safe_call(df.set_plane_task, name, TASK_LABEL_DEAD)
                    self.applied[name] = {"plane": name, "task": "_dead"}
                continue
            if a == self.applied.get(name):
                continue  # unchanged -> no command spam
            task = a["task"]
            if task == "engage":
                # set the target BEFORE activating the IA: activation re-targets
                # randomly when no target is selected.
                self.safe_call(df.set_target_id, name, a.get("target") or 0)
                self.safe_call(df.activate_IA, name)
            elif task in ("patrol", "retreat"):
                self.safe_call(df.deactivate_IA, name)
                self.safe_call(df.stabilize_plane, name)
                self.safe_call(df.activate_autopilot, name)
                if a.get("heading") is not None:
                    self.safe_call(df.set_plane_autopilot_heading, name, a["heading"])
                if a.get("altitude") is not None:
                    self.safe_call(df.set_plane_autopilot_altitude, name, a["altitude"])
                if a.get("speed") is not None:
                    self.safe_call(df.set_plane_autopilot_speed, name, a["speed"])
            # hold: leave everything as is
            label = self.label_for(a)
            self.safe_call(df.set_plane_task, name, label)
            self.applied[name] = a
            print("[commander] %-10s -> %-8s %s" % (name, task, a.get("target") or ""))
        # planes that left the assignment list: dead -> one-shot "坠毁" label,
        # otherwise just forget them (e.g. side changed)
        for name in self.own:
            if name in live_own:
                continue
            if not tactician.plane_alive(states.get(name)):
                if self.applied.get(name, {}).get("task") != "_dead":
                    self.safe_call(df.set_plane_task, name, TASK_LABEL_DEAD)
                self.applied[name] = {"plane": name, "task": "_dead"}
            else:
                self.applied.pop(name, None)

    @staticmethod
    def label_for(a):
        base = TASK_LABELS.get(a["task"], a["task"])
        if a["task"] == "engage" and a.get("target"):
            return u"%s %s" % (base, a["target"])
        return base
    def log_decision(self, sim_time, result):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(sim_time, 1), "engine": self.engine.name,
                                    "reason": result.get("reason", ""),
                                    "assignments": result.get("assignments", [])},
                                   ensure_ascii=False) + "\n")
        except OSError as exc:
            print("[commander] log write failed: %r" % (exc,))

    def safe_call(self, fn, *args):
        if self.dry_run:
            return
        try:
            fn(*args)
        except Exception as exc:
            print("[commander] command failed: %r(%r): %r" % (getattr(fn, "__name__", fn), args, exc))

    # ------------------------------------------------------------------ loop

    def run(self, duration_s=None, once=False):
        self.connect()
        self.missiles_t = -1e9
        start = time.time()
        next_decide = 0.0  # decide immediately after connect
        try:
            while True:
                now = time.time() - start
                if duration_s is not None and now > duration_s:
                    break
                states = self.poll_states()
                self.poll_missiles(now)

                alive_sig = tuple(sorted(n for n in self.planes if tactician.plane_alive(states.get(n))))
                event = alive_sig != self.last_alive_sig
                due = now >= next_decide
                if states and (due or (event and self.last_alive_sig is not None)):
                    self.decide(states)
                    self.last_decision_t = now
                    next_decide = now + self.period_s
                self.last_alive_sig = alive_sig
                if once:
                    break
                time.sleep(self.poll_s)
        except KeyboardInterrupt:
            print("\n[commander] interrupted")
        finally:
            try:
                df.disconnect()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="dogfight mission commander")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--dry-run", action="store_true", help="print decisions, do not send commands")
    ap.add_argument("--once", action="store_true", help="single decision cycle then exit")
    ap.add_argument("--duration", type=float, default=None, help="stop after N wall seconds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cmd = Commander(cfg, dry_run=args.dry_run)
    cmd.run(duration_s=args.duration, once=args.once)


if __name__ == "__main__":
    main()
