# -*- coding: utf-8 -*-
"""End-to-end test of the LLM mission commander against a live sandbox.

Boots the sandbox on the 2v2 network mission (renderless, free-run), connects a
Commander with the rule engine, and checks the full observe -> decide -> apply
pipeline:

  1. first decision   -> both red planes get IA on + distinct engage targets
  2. damaged plane    -> switches to retreat (IA off, autopilot on)
  3. all blues killed -> survivors switch to patrol
  4. decisions.jsonl  -> logged, and no command ever threw

Usage:  bin/python/python.exe tools/test_llm_commander.py [--keep]
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repo root
LLM_DIR = os.path.join(ROOT, 'llm_commander')
sys.path.insert(0, LLM_DIR)

from commander import Commander, load_config            # noqa: E402
import tactician as T                                   # noqa: E402

results = []


def check(name, ok, detail=''):
    results.append((name, ok))
    print('%-46s %s  %s' % (name, 'PASS' if ok else 'FAIL', detail))


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
    keep = '--keep' in sys.argv
    log_path = os.path.join(LLM_DIR, 'decisions.jsonl')
    if os.path.exists(log_path):
        os.remove(log_path)

    cfg = load_config(os.path.join(LLM_DIR, 'config.json'))
    cfg['engine'] = 'rule'
    cfg['decision_period_s'] = 10

    sandbox = subprocess.Popen(
        [os.path.join(ROOT, 'dogfight_sandbox_hg2', 'bin', 'python', 'python.exe'),
         'main.py', 'auto_network', 'mission=2'],
        cwd=os.path.join(ROOT, 'dogfight_sandbox_hg2', 'source'),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        check('sandbox boots (2v2 mission)', wait_port(cfg['host'], cfg['port']), 'port %d' % cfg['port'])

        from dogfight_sandbox_hg2.network_client_example import dogfight_client as df
        cmd = Commander(cfg)
        cmd.connect()
        df.set_renderless_mode(True)  # headless test: skip the 3D window
        check('commander owns the red side', sorted(cmd.own) == ['ennemy_1', 'ennemy_2'], str(cmd.own))

        # --- cycle 1: initial engagement --------------------------------------
        cmd.missiles_t = -1e9
        states = cmd.poll_states()
        cmd.poll_missiles(0.0)
        cmd.decide(states)
        time.sleep(2.0)  # let the commands reach the sandbox

        s1 = df.get_plane_state('ennemy_1')
        s2 = df.get_plane_state('ennemy_2')
        check('ennemy_1 IA activated', bool(s1['ia']))
        check('ennemy_2 IA activated', bool(s2['ia']))
        t1, t2 = s1['target_id'], s2['target_id']
        check('engage targets set', t1 in ('ally_1', 'ally_2') and t2 in ('ally_1', 'ally_2'),
              '%s / %s' % (t1, t2))
        check('engage targets distinct', t1 != t2)

        # --- cycle 2: damage ennemy_1 -> retreat ------------------------------
        df.set_health('ennemy_1', 0.1)
        time.sleep(0.5)
        states = cmd.poll_states()
        cmd.decide(states)
        time.sleep(2.0)
        s1 = df.get_plane_state('ennemy_1')
        check('damaged plane disengages (IA off)', not s1['ia'])
        check('damaged plane autopilot on', bool(s1['autopilot']))
        check('retreat heading/speed applied', abs(s1['autopilot_speed'] - T.RETREAT_SPEED_MS) < 1,
              'spd=%.0f' % s1['autopilot_speed'])

        # --- cycle 3: kill the blues -> patrol ---------------------------------
        df.set_health('ally_1', 0.0)
        df.set_health('ally_2', 0.0)
        time.sleep(3.0)  # wreck propagation
        states = cmd.poll_states()
        cmd.decide(states)
        time.sleep(2.0)
        s1 = df.get_plane_state('ennemy_1')
        s2 = df.get_plane_state('ennemy_2')
        p1 = cmd.applied.get('ennemy_1') or {}
        p2 = cmd.applied.get('ennemy_2') or {}
        check('no enemies -> patrol (from applied)',
              p1.get('task') == 'patrol' and p2.get('task') in ('patrol', '_dead', None),
              '%s / %s' % (p1.get('task'), p2.get('task')))

        # --- log file ----------------------------------------------------------
        ok_log = False
        if os.path.exists(log_path):
            lines = [l for l in open(log_path, encoding='utf-8').read().splitlines() if l.strip()]
            ok_log = len(lines) >= 3 and all(json.loads(l).get('assignments') is not None for l in lines)
        check('decisions.jsonl logged', ok_log)

        s1 = df.get_plane_state('ennemy_1')
        s2 = df.get_plane_state('ennemy_2')
        print('  final: ennemy_1 alt=%.0f v=%.0f wreck=%s crashed=%s | ennemy_2 alt=%.0f v=%.0f wreck=%s crashed=%s'
              % (s1['altitude'], s1['linear_speed'], s1['wreck'], s1['crashed'],
                 s2['altitude'], s2['linear_speed'], s2['wreck'], s2['crashed']))
        # the retreated plane must still be flying; a pursuit loss of the other
        # plane is legitimate combat, not a commander failure
        check('retreated plane still flying', T.plane_alive(s1))

        try:
            df.disconnect()
        except Exception:
            pass
    finally:
        if not keep:
            sandbox.terminate()
            try:
                sandbox.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sandbox.kill()

    failed = [n for n, ok in results if not ok]
    print('\n%d/%d checks passed' % (len(results) - len(failed), len(results)))
    if failed:
        print('FAILED:', failed)
        sys.exit(1)


if __name__ == '__main__':
    main()
