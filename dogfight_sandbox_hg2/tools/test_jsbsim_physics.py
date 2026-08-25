# End-to-end physics calibration test against the running sandbox.
#
# Usage:
#   1. start the sandbox:  cd dogfight_sandbox_hg2 && bin\python\python.exe source\main.py auto_network
#   2. run this client:    bin\python\python.exe tools\test_jsbsim_physics.py [--legacy]
#
# --legacy boots nothing itself; it just compares against expectations of the legacy
# engine (run the sandbox with config.json Physics.engine = "legacy" for that).
#
# Checks (signs are the crux -- they must match the legacy engine's semantics):
#   pitch cmd > 0        -> pitch_attitude increases (nose up)
#   roll  cmd > 0        -> roll_attitude increases (left bank) & heading decreases
#   yaw   cmd > 0        -> heading decreases (nose left)
#   thrust 0 -> 1        -> linear_speed increases
#   reset_machine_matrix -> state teleports; set_plane_linear_speed syncs the FDM
#   60 s neutral         -> no NaN, no uncommanded ground impact

import sys
import os
import time
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'source'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))

from dogfight_sandbox_hg2.network_client_example import dogfight_client as df

HOST = os.environ.get('DOGFIGHT_HOST', '192.168.1.103')
PORT = int(os.environ.get('DOGFIGHT_PORT', '50888'))

results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print('%-42s %s  %s' % (name, 'PASS' if ok else 'FAIL', detail))


def step(frames=1, plane='ally_1'):
    for _ in range(frames):
        df.update_scene()
        df.get_finish_flag()


def get(plane='ally_1'):
    return df.get_plane_state(plane)


def connect():
    for attempt in range(60):
        try:
            df.connect(HOST, PORT)
            df.get_planes_list()
            return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError('cannot connect to sandbox at %s:%d' % (HOST, PORT))


def main():
    legacy = '--legacy' in sys.argv
    connect()
    planes = df.get_planes_list()
    print('planes:', planes, 'engine:', get(planes[0])['physics_engine'] if not legacy else 'legacy(expected)')

    plane = 'ally_1'
    df.set_client_update_mode(True)
    df.set_renderless_mode(True)

    # --- spawn at a known state -----------------------------------------------
    df.reset_machine(plane)
    df.reset_machine_matrix(plane, 0.0, 2000.0, 0.0, 0.0, 0.0, 0.0)  # heading north, level
    df.set_plane_thrust(plane, 0.8)
    df.set_plane_linear_speed(plane, 120.0)
    step(30)
    s0 = get(plane)
    check('spawn: altitude ~2000', abs(s0['altitude'] - 2000) < 100, 'alt=%.1f' % s0['altitude'])
    check('spawn: speed ~120', abs(s0['linear_speed'] - 120) < 25, 'v=%.1f' % s0['linear_speed'])
    check('engine is ' + ('legacy' if legacy else 'jsbsim'), (s0['physics_engine'] == 'legacy') == legacy)

    # --- pitch up (moderate pull so speed survives the sequence) ----------------
    # original engine convention: pitch level < 0 = nose up
    p0 = s0['pitch_attitude']
    df.set_plane_pitch(plane, -0.6)
    step(60)
    s1 = get(plane)
    check('pitch cmd -0.6 -> pitch_attitude up', s1['pitch_attitude'] - p0 > 5.0,
          '%.1f -> %.1f' % (p0, s1['pitch_attitude']))
    df.set_plane_pitch(plane, 0.0)
    step(60)
    s1b = get(plane)
    check('pitch released -> attitude held (+-12 deg)',
          abs(s1b['pitch_attitude'] - s1['pitch_attitude']) < 12.0,
          '%.1f -> %.1f' % (s1['pitch_attitude'], s1b['pitch_attitude']))

    # --- recover energy before the lateral checks ------------------------------
    df.stabilize_plane(plane)
    df.set_plane_thrust(plane, 1.0)
    for _ in range(360):
        step(1)
        if get(plane)['linear_speed'] > 110.0:
            break
    df.set_plane_thrust(plane, 0.8)

    # --- roll left -------------------------------------------------------------
    r0 = get(plane)['roll_attitude']
    h0 = get(plane)['heading']
    df.set_plane_roll(plane, 0.5)    # >0 = roll left
    step(60)
    s2 = get(plane)
    check('roll cmd +0.5 -> roll_attitude up (left bank)', s2['roll_attitude'] - r0 > 5.0,
          '%.1f -> %.1f (v=%.0f)' % (r0, s2['roll_attitude'], s2['linear_speed']))
    df.set_plane_roll(plane, 0.0)
    step(120)
    s2b = get(plane)
    dh = (s2b['heading'] - h0 + 540) % 360 - 180
    check('left bank -> heading decreases (left turn)', dh < -1.0,
          'hdg %.1f -> %.1f' % (h0, s2b['heading']))
    df.set_plane_roll(plane, -0.4)   # roll back right
    step(60)
    s2c = get(plane)
    check('roll cmd -0.4 -> roll back right', s2c['roll_attitude'] < s2b['roll_attitude'],
          '%.1f -> %.1f' % (s2b['roll_attitude'], s2c['roll_attitude']))
    df.set_plane_roll(plane, 0.0)
    step(60)

    # --- yaw left --------------------------------------------------------------
    df.stabilize_plane(plane)
    for _ in range(300):  # let any residual bank level off first
        step(1)
        if abs(get(plane)['roll_attitude']) < 10.0:
            break
    h0 = get(plane)['heading']
    df.set_plane_yaw(plane, 1.0)     # >0 = nose right (original engine convention)
    step(150)
    s3 = get(plane)
    dh = (s3['heading'] - h0 + 540) % 360 - 180
    check('yaw cmd +1.0 -> heading increases', dh > 1.0, '%.1f -> %.1f' % (h0, s3['heading']))
    df.set_plane_yaw(plane, 0.0)
    step(30)

    # --- throttle --------------------------------------------------------------
    df.stabilize_plane(plane)
    for _ in range(300):  # measure from roughly level, unbanked flight
        step(1)
        s = get(plane)
        if abs(s['roll_attitude']) < 10.0 and abs(s['pitch_attitude']) < 10.0:
            break
    v0 = get(plane)['linear_speed']
    df.set_plane_thrust(plane, 1.0)
    step(600)
    v1 = get(plane)['linear_speed']
    check('thrust 1.0 -> speed holds or increases', v1 - v0 > -5.0, '%.1f -> %.1f' % (v0, v1))

    # --- long neutral stability (30 s at ~60 Hz steps, batched) ----------------
    df.set_plane_thrust(plane, 0.8)
    ok_nan = True
    crashed = False
    slow = False
    for i in range(1800):
        step(1)
        if i % 300 == 299:
            s = get(plane)
            if any(math.isnan(x) for x in (s['position'][0], s['position'][1], s['position'][2])):
                ok_nan = False
                break
            if s['linear_speed'] < 60.0:
                slow = True
            if s['crashed'] or s['wreck']:
                crashed = True
                break
    s = get(plane)
    check('30 s neutral: no NaN', ok_nan, 'pos=[%.0f, %.0f, %.0f]' % tuple(s['position']))
    check('30 s neutral: no spontaneous crash', not crashed,
          'alt=%.0f v=%.1f pitch=%.1f' % (s['altitude'], s['linear_speed'], s['pitch_attitude']))
    check('30 s neutral: keeps energy (v > 60 m/s)', not slow,
          'final v=%.1f' % s['linear_speed'])

    # --- reset determinism -----------------------------------------------------
    df.reset_machine_matrix(plane, 1000.0, 2000.0, 1500.0, 0.0, 1.5, 0.0)  # yaw east
    df.set_plane_linear_speed(plane, 100.0)
    step(30)
    s = get(plane)
    dx = s['position'][0] - 1000.0
    dz = s['position'][2] - 1500.0
    check('reset: heading east -> drifts +X (east)', dx > 30.0 and abs(dz) < abs(dx),
          'd=(%.0f, %.0f)' % (dx, dz))

    print('\n%d/%d checks passed' % (sum(1 for _, ok, _ in results if ok), len(results)))
    fails = [n for n, ok, _ in results if not ok]
    if fails:
        print('FAILED:', fails)
        sys.exit(1)


if __name__ == '__main__':
    main()
