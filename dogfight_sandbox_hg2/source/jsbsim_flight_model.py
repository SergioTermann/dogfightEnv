# JSBSim six-degree-of-freedom flight dynamics wrapper for the dogfight sandbox.
#
# Replaces the simplified point-mass model in Physics.py for AIRCRAFT only
# (missiles keep the legacy proportional-navigation model).
#
# Coordinate conventions
#   Harfang world : x = East, y = Up (meters, ASL), z = North; heading 0 = +Z, 90 = +X (compass, cw).
#   JSBSim        : geodetic lat/lon/alt ASL; Euler psi (heading cw from north), theta (pitch up +), phi (roll right bank +).
#
# Property quirks of the bundled jsbsim 1.2.1 build (verified empirically):
#   - velocities/{p,q,r}-{rad,deg}-sec, accelerations/n-pilot-z-normal and position/h-sl-m are
#     present in the catalog but never update -> angular rates are obtained by finite
#     differences of the Euler angles and world velocity by differentiating position.
#   - attitude/theta-deg, psi-deg, phi-deg, aero/alpha-deg, aero/beta-deg,
#     velocities/vc-kts, position/h-sl-ft, position/h-agl-ft, position/lat-gc-deg,
#     position/long-gc-deg are live and used here.
#
# The stock f16.xml has no stability augmentation (the real airframe is statically
# unstable), so a small rate-command SAS is layered on top: the sandbox stick
# commands (angular_levels, -1..1) are interpreted as rate commands around the
# trimmed state, which keeps the control semantics of the original sandbox.

import contextlib
import io
import math
import os

import harfang as hg

try:
    import jsbsim
except ImportError:  # sandbox runs without the jsbsim pylibs -> stay on legacy physics
    jsbsim = None

M_TO_FT = 1.0 / 0.3048
FT_TO_M = 0.3048
KT_TO_MS = 0.514444
MS_TO_KT = 1.0 / KT_TO_MS
R_EARTH = 6371000.0

# model name -> jsbsim aircraft dir (all fighters currently fly F-16 data)
AIRCRAFT_MODEL_MAP = {
    'F16': 'f16',
    'Rafale': 'f16',
    'Eurofighter': 'f16',
    'F14': 'f16',
    'F14_2': 'f16',
    'TFX': 'f16',
    'Miuss': 'f16',
}

# --- control conventions (matching the ORIGINAL engine semantics, see MachineDevice
# autopilot + client_sample: pitch level < 0 = nose up; yaw level > 0 = nose right;
# roll level > 0 = roll left. jsbsim f16.xml: elevator-cmd > 0 = nose down,
# aileron-cmd > 0 = roll right, rudder-cmd > 0 = nose left.

# --- SAS gains ---
# f16.xml control conventions (empirical): elevator-cmd > 0 = nose down, aileron-cmd > 0 = roll right,
# rudder-cmd > 0 = nose left.
#
# Pitch channel: load-compensated ALPHA HOLD (like a real FLCS). A neutral stick holds a
# small target AoA (scaled by 1/cos(bank) so banked turns keep altitude), which makes the
# aircraft seek its natural cruise speed instead of mushing at high alpha. The stick
# integrates an AoA offset on top; releasing it decays back to the trim AoA.
ALPHA_TARGET_DEG = 3.5    # trim AoA at wings level (speed-seeking equilibrium)
GAMMA_ALPHA = 0.45        # deg of target-AoA UNLOAD per deg of sustained climb: excess
                         # thrust turns into speed instead of an endless zoom climb
ALPHA_CMD_RATE = 20.0     # deg/s of AoA command per unit stick
ALPHA_CMD_MIN = -6.0      # AoA command clamp (push)
ALPHA_CMD_MAX = 19.0      # AoA command clamp (pull, below stall)
ALPHA_DECAY = 0.35        # 1/s: AoA offset decays when the pitch stick is neutral
K_ALPHA = 0.08            # elevator per deg of AoA error
K_Q = 1.6                 # elevator per rad/s pitch rate (the f16.xml FCS also damps)
K_TRIM = 0.006            # slow integral on AoA error (steady-state elevator)
TRIM_MAX = 0.3            # elevator trim integrator clamp
GAMMA_DAMP_DEG = 10.0     # flight-path damping deadband: zooms beyond this are opposed
K_GAMMA = 0.02            # elevator per deg of flight-path angle beyond the deadband
GCAS_AGL_M = 250.0        # auto ground-collision avoidance (real F-16 Block 40+ feature;
GCAS_PREDICT_S = 2.5      # also protects the legacy-tuned IA). Fires near the ground OR
                         # when the current sink rate reaches the ground within the horizon.
ALPHA_LIMIT_DEG = 24.0    # hard AoA protection above this alpha
ALPHA_PROTECT = 0.06      # elevator per degree over the limit
ROLL_CMD_RATE = 120.0    # deg/s of roll command per unit stick
ROLL_CMD_MAX = 175.0     # deg
K_PHI = 0.03             # aileron per deg of bank error
K_P = 0.12               # aileron per rad/s roll rate
YAW_FF = 0.3             # rudder per unit stick (f16's internal FLCS heavily damps it)
YAW_ROLL_RATE = 90.0     # deg/s of bank per unit yaw stick: the f16 turns by banking, so the
                         # yaw command also banks the aircraft (easy-steering compatible)
LEVEL_DECAY = 0.35       # 1/s: bank command decays toward level when sticks are neutral
K_R = 0.8                # rudder per rad/s yaw rate (after turn-rate feedforward)
K_BETA = 0.01            # rudder per deg of sideslip (coordination)

MIN_AIRSPEED_MS = 85.0  # spawn/fallback airspeed when airborne with no speed
MIN_CONTROLLABLE_ALT_M = 50.0

# Global switch, set from config.json "Physics" -> {"engine": "jsbsim"|"legacy"} by main.py
USE_JSBSIM = True
DEBUG = False  # per-frame diagnostics to stdout

# Flight-path prediction overlay (config.json "FlightPrediction"): a kinematic
# rollout of the current state drawn in the 3D view by Main.display_flight_prediction.
PREDICT_ENABLED = True
PREDICT_HORIZON_S = 10.0
PREDICT_STEPS = 20


def configure(physics_config):
    """Apply the config.json "Physics" section (engine, origin)."""
    global USE_JSBSIM
    if not physics_config:
        return
    USE_JSBSIM = str(physics_config.get('engine', 'jsbsim')).lower() == 'jsbsim'


def configure_prediction(prediction_config):
    """Apply the config.json "FlightPrediction" section."""
    global PREDICT_ENABLED, PREDICT_HORIZON_S, PREDICT_STEPS
    if not prediction_config:
        return
    PREDICT_ENABLED = bool(prediction_config.get('enabled', True))
    PREDICT_HORIZON_S = float(prediction_config.get('horizon_s', 10.0))
    PREDICT_STEPS = max(4, int(prediction_config.get('steps', 20)))


def jsbsim_available():
    return jsbsim is not None


class JSBSimFlightModel:
    """One instance per aircraft; owns an FGFDMExec and converts states both ways."""

    FDM_DT = 1.0 / 60.0

    def __init__(self, sandbox_type='F16', origin_lat=0.0, origin_lon=0.0):
        self.model = AIRCRAFT_MODEL_MAP.get(sandbox_type, 'f16')
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        with contextlib.redirect_stdout(io.StringIO()):  # keep the sandbox console clean
            self.fdm = jsbsim.FGFDMExec(os.path.dirname(jsbsim.__file__))
            self.fdm.set_debug_level(0)
            self.fdm.load_model(self.model)
            self.fdm.set_dt(self.FDM_DT)
        self._prev_euler = (0.0, 0.0, 0.0)   # (theta, psi, phi) deg, for rate differences
        self._prev_pos = hg.Vec3(0, 0, 0)     # world meters
        self._last_speed_ms = 0.0             # for flight-path angle (speed protection)
        self._last_vy = 0.0
        self._alpha_cmd_deg = 0.0             # SAS AoA offset (integrated from pitch stick)
        self._roll_cmd_deg = 0.0
        self._elev_trim = 0.0                 # slow integrator canceling the untrimmed moment
        self._prev_speed_ms = 0.0             # for the acceleration estimate
        self._last_rates = (0.0, 0.0, 0.0)    # (q, p, r) rad/s, measured across the last FDM step
        self._pred_state = None               # cached state for predict_path()
        self._ready = False

    # ------------------------------------------------------------------ helpers

    def _set_if_exists(self, prop, value):
        try:
            self.fdm.set_property_value(prop, value)
            return True
        except Exception:
            return False

    def _world_from_geodetic(self, lat_deg, lon_deg, alt_m):
        north = math.radians(lat_deg - self.origin_lat) * R_EARTH
        east = math.radians(lon_deg - self.origin_lon) * R_EARTH * math.cos(math.radians(self.origin_lat))
        return hg.Vec3(east, alt_m, north)

    def _geodetic_from_world(self, pos):
        lat = self.origin_lat + math.degrees(pos.z / R_EARTH)
        lon = self.origin_lon + math.degrees(pos.x / (R_EARTH * math.cos(math.radians(self.origin_lat))))
        return lat, lon, pos.y

    @staticmethod
    def _attitudes_from_matrix(matrix):
        """Same math as the legacy Physics.update_physics: heading/pitch/roll in degrees."""
        aX, aY, aZ = hg.GetX(matrix), hg.GetY(matrix), hg.GetZ(matrix)
        y_dir = 1.0 if aY.y > 0 else -1.0
        horizontal_aZ = hg.Normalize(hg.Vec3(aZ.x, 0, aZ.z))
        horizontal_aX = hg.Cross(hg.Vec3.Up, horizontal_aZ) * y_dir
        pitch = math.degrees(math.acos(max(-1.0, min(1.0, hg.Dot(horizontal_aZ, aZ)))))
        if aZ.y < 0:
            pitch = -pitch
        roll = math.degrees(math.acos(max(-1.0, min(1.0, hg.Dot(horizontal_aX, aX)))))
        if aX.y < 0:
            roll = -roll
        heading = math.degrees(math.acos(max(-1.0, min(1.0, hg.Dot(horizontal_aZ, hg.Vec3.Front)))))
        if horizontal_aZ.x < 0:
            heading = 360.0 - heading
        return heading, pitch, roll

    @staticmethod
    def _matrix_from_attitudes(heading_deg, pitch_deg, roll_deg, pos):
        """Build a Harfang world matrix so that GetR returns (-pitch, heading, -roll) in radians,
        matching the convention of matrices produced by the legacy physics (rx>0 nose down, rz>0 roll left)."""
        psi = math.radians(heading_deg)
        rx = math.radians(-pitch_deg)
        rz = math.radians(-roll_deg)
        rot = hg.RotationMatY(psi) * hg.RotationMatX(rx) * hg.RotationMatZ(rz)
        return hg.TransformationMat4(pos, rot)

    def _refuel(self):
        # note: tank[0] is unindexed in this build's property tree, and the capacity
        # properties do not exist (reading them yields 0 -- emptying the tanks and
        # flaming the engine out), so fill with a fixed realistic load instead:
        # 4 x 1750 lbs = 7000 lbs internal fuel.
        for name in ('propulsion/tank', 'propulsion/tank[1]', 'propulsion/tank[2]', 'propulsion/tank[3]'):
            try:
                self.fdm.set_property_value(name + '/contents-lbs', 1750.0)
            except Exception:
                pass

    # ------------------------------------------------------------------ state sync

    def sync_from_kinematics(self, matrix, v_move, thrust_level=0.8):
        """(Re)initialize the FDM from sandbox state. Called on spawn/reset/teleport and
        whenever the sandbox externally overwrites matrix or velocity."""
        heading, pitch, roll = self._attitudes_from_matrix(matrix)
        pos = hg.GetT(matrix)
        speed = hg.Len(v_move)
        # Only rescue AIRBORNE spawns with the fallback airspeed when the engine is
        # actually commanded on; pre-mission idle aircraft (speed 0, throttle 0) sink
        # in place like the legacy engine until the client resets them into flight.
        if speed < MIN_AIRSPEED_MS and pos.y > MIN_CONTROLLABLE_ALT_M and thrust_level > 0.05:
            speed = MIN_AIRSPEED_MS
        gamma = 0.0
        if hg.Len(v_move) > 1.0:
            gamma = math.degrees(math.asin(max(-1.0, min(1.0, v_move.y / hg.Len(v_move)))))

        lat, lon, alt = self._geodetic_from_world(pos)
        with contextlib.redirect_stdout(io.StringIO()):
            # order matters: speed ICs first -- setting them re-derives attitude from alpha,
            # so psi/theta/phi must be applied last to stick (see tools/test_jsbsim_physics.py)
            self.fdm.set_property_value('ic/lat-gc-deg', lat)
            self.fdm.set_property_value('ic/long-gc-deg', lon)
            self.fdm.set_property_value('ic/h-sl-ft', alt * M_TO_FT)
            self.fdm.set_property_value('ic/vc-kts', speed * MS_TO_KT)
            self.fdm.set_property_value('ic/gamma-deg', gamma)
            self.fdm.set_property_value('ic/psi-true-deg', heading)
            self.fdm.set_property_value('ic/theta-deg', pitch)
            self.fdm.set_property_value('ic/phi-deg', roll)
            self.fdm.run_ic()

            self._refuel()
            # engine start needs the INDEXED name in this build; the fcs/ throttle bus is unindexed
            self._set_if_exists('propulsion/engine[0]/set-running', 1)
            self.fdm.set_property_value('fcs/throttle-cmd-norm', 0.5 * max(0.0, min(1.0, thrust_level)))
            self.fdm.set_property_value('fcs/elevator-cmd-norm', 0.0)
            self.fdm.set_property_value('fcs/aileron-cmd-norm', 0.0)
            self.fdm.set_property_value('fcs/rudder-cmd-norm', 0.0)

        self._prev_euler = (pitch, heading, roll)
        self._alpha_cmd_deg = 0.0
        self._roll_cmd_deg = roll
        self._elev_trim = 0.0
        self._last_rates = (0.0, 0.0, 0.0)  # state jump invalidates rate estimates
        self._last_speed_ms = speed
        self._last_vy = hg.Len(v_move) * math.sin(math.radians(gamma)) if hg.Len(v_move) > 1.0 else 0.0
        self._prev_pos = self._current_world_position()
        self._ready = True

    def _current_world_position(self):
        lat = float(self.fdm.get_property_value('position/lat-gc-deg'))
        lon = float(self.fdm.get_property_value('position/long-gc-deg'))
        alt = float(self.fdm.get_property_value('position/h-sl-ft')) * FT_TO_M
        return self._world_from_geodetic(lat, lon, alt)

    # ------------------------------------------------------------------ trajectory prediction

    def predict_path(self, horizon_s=10.0, steps=20):
        """Kinematic rollout of the current flight state for the HUD overlay.

        Propagates position/speed/heading/flight-path with the CURRENT turn rate,
        pitch rate and acceleration, each decaying exponentially (a held turn
        keeps turning, but the model doesn't pretend to know future stick inputs).
        Returns world-space points, starting at the current position."""
        if not self._ready or self._pred_state is None:
            return []
        pos, speed, hdg, theta, phi, turn_r, pitch_r, accel = self._pred_state
        alpha_deg = float(self.fdm.get_property_value('aero/alpha-deg'))
        gamma0 = max(-80.0, min(80.0, theta - alpha_deg))
        tau_turn, tau_pitch, tau_acc = 4.0, 1.5, 3.0
        step = horizon_s / steps

        def decayed_integral(rate, tau, t0, t1):
            return rate * tau * (math.exp(-t0 / tau) - math.exp(-t1 / tau))

        pts = [hg.Vec3(pos.x, pos.y, pos.z)]
        p = pts[0]
        gamma = gamma0
        for i in range(steps):
            t0, t1 = i * step, (i + 1) * step
            hdg_i = hdg + turn_r * tau_turn * (1.0 - math.exp(-t1 / tau_turn))
            gamma = max(-70.0, min(70.0, gamma0 + pitch_r * tau_pitch * (1.0 - math.exp(-t1 / tau_pitch))))
            speed_i = max(40.0, speed + accel * tau_acc * (1.0 - math.exp(-t1 / tau_acc)))
            g, h = math.radians(gamma), math.radians(hdg_i)
            v = hg.Vec3(math.cos(g) * math.sin(h), math.sin(g), math.cos(g) * math.cos(h)) * (speed_i * step)
            p = p + v
            if p.y < 1.0:  # predicted ground contact: run level from here on
                p.y = 1.0
                gamma0 = 0.0
                gamma = 0.0
            pts.append(hg.Vec3(p.x, p.y, p.z))
        return pts

    # ------------------------------------------------------------------ per-frame update

    def update(self, matrix, physics_parameters, dts):
        """Advance the FDM one fixed step and return (matrix, out_params) in the legacy contract."""
        if not self._ready:
            self.sync_from_kinematics(matrix, physics_parameters.get('v_move', hg.Vec3(0, 0, 0)),
                                      physics_parameters.get('thrust_level', 0.8))

        thrust_level = physics_parameters.get('thrust_level', 0.0)
        angular = physics_parameters.get('angular_levels', hg.Vec3(0, 0, 0))
        wreck_factor = physics_parameters.get('health_wreck_factor', 1.0)
        authority = wreck_factor  # damage reduces control authority

        # --- rate estimates: measured across the LAST FDM step (post-run vs pre-run of
        # the previous frame); a one-frame lag is negligible for damping at 60 Hz ---
        theta = self.fdm.get_property_value('attitude/theta-deg')
        psi = self.fdm.get_property_value('attitude/psi-deg')
        phi = self.fdm.get_property_value('attitude/phi-deg')
        theta0, psi0, phi0 = theta, psi, phi
        q_rate, p_rate, r_rate = self._last_rates
        dt = self.FDM_DT

        alpha = self.fdm.get_property_value('aero/alpha-deg')
        beta = self.fdm.get_property_value('aero/beta-deg')

        # --- SAS: integrate sticks into AoA / bank commands, then hold them ---
        # pitch level < 0 = nose up (original engine convention), so the AoA offset
        # integrates with -=; roll level > 0 = roll left (phi decreases) also uses -=.
        self._alpha_cmd_deg -= ALPHA_CMD_RATE * angular.x * authority * dt
        if abs(angular.x) < 0.05:
            self._alpha_cmd_deg *= max(0.0, 1.0 - dt * ALPHA_DECAY)
        # yaw level > 0 = nose right: banks right (phi increases) via +=.
        self._roll_cmd_deg = max(-ROLL_CMD_MAX, min(ROLL_CMD_MAX,
                                    self._roll_cmd_deg - (ROLL_CMD_RATE * angular.z
                                                          - YAW_ROLL_RATE * angular.y) * authority * dt))
        # easy-steering behavior (like the legacy engine): with lateral sticks neutral
        # the bank command eases back toward wings-level, preventing locked-in spiral dives.
        if abs(angular.z) < 0.05 and abs(angular.y) < 0.05:
            self._roll_cmd_deg *= max(0.0, 1.0 - dt * LEVEL_DECAY)

        # pitch channel: hold AoA (load-compensated by 1/cos(bank) so turns keep altitude);
        # a sustained climb UNLOADS the AoA target so excess thrust becomes speed
        # instead of an endless zoom climb.
        gamma_deg = None
        if self._last_speed_ms > 30.0:
            gamma_deg = math.degrees(math.asin(max(-1.0, min(1.0, self._last_vy / self._last_speed_ms))))
        alpha_base = ALPHA_TARGET_DEG / max(0.35, math.cos(math.radians(phi)))
        if gamma_deg is not None:
            alpha_base -= GAMMA_ALPHA * max(0.0, min(gamma_deg, 15.0))
        alpha_cmd = alpha_base + self._alpha_cmd_deg
        alpha_cmd = max(ALPHA_CMD_MIN, min(ALPHA_CMD_MAX, alpha_cmd))

        # --- auto-GCAS: sinking fast near the ground (gear up) -> max pull, wings level ---
        terrain_alt = physics_parameters.get('terrain_altitude_m')
        ground = terrain_alt if terrain_alt is not None else 0.0
        agl = self._prev_pos.y - ground
        predicted_agl = agl + self._last_vy * GCAS_PREDICT_S
        if (self._last_vy < -5.0 and (agl < GCAS_AGL_M or predicted_agl < GCAS_AGL_M)
                and self._last_speed_ms > 60.0 and not physics_parameters.get('gear_down')):
            alpha_cmd = ALPHA_CMD_MAX
            self._roll_cmd_deg *= max(0.0, 1.0 - dt * 6.0)

        alpha_err = alpha - alpha_cmd
        saturated = alpha_cmd <= ALPHA_CMD_MIN + 0.01 or alpha_cmd >= ALPHA_CMD_MAX - 0.01
        if not saturated:  # anti-windup: trim integrator only runs when unclamped
            self._elev_trim = max(-TRIM_MAX, min(TRIM_MAX, self._elev_trim + K_TRIM * alpha_err * dt))
        elevator = K_ALPHA * alpha_err + K_Q * q_rate + self._elev_trim
        # phugoid damping: oppose large flight-path excursions (zoom climbs / mush sinks)
        # while leaving energy-maneuvering inside the deadband untouched.
        if gamma_deg is not None:
            if gamma_deg > GAMMA_DAMP_DEG:
                elevator += K_GAMMA * (gamma_deg - GAMMA_DAMP_DEG)
            elif gamma_deg < -GAMMA_DAMP_DEG:
                elevator += K_GAMMA * (gamma_deg + GAMMA_DAMP_DEG)
        if alpha > ALPHA_LIMIT_DEG:
            elevator += ALPHA_PROTECT * (alpha - ALPHA_LIMIT_DEG)
        elif alpha < -10.0:
            elevator += ALPHA_PROTECT * (-10.0 - alpha)

        aileron = -K_PHI * (phi - self._roll_cmd_deg) - K_P * p_rate  # aileron + = roll right (phi +)

        # expected turn rate for a coordinated bank (deg/s) so the damper doesn't fight it
        speed = max(self.fdm.get_property_value('velocities/vc-kts') * KT_TO_MS, 50.0)
        turn_rate_deg_s = math.degrees(9.81 * math.tan(math.radians(phi)) / speed) if abs(phi) < 80 else 0.0
        # yaw level > 0 = nose right -> rudder-cmd negative (f16: positive = nose left)
        rudder = -YAW_FF * angular.y * authority + K_R * (r_rate - math.radians(turn_rate_deg_s)) - K_BETA * beta

        self.fdm.set_property_value('fcs/elevator-cmd-norm', max(-1.0, min(1.0, elevator)))
        self.fdm.set_property_value('fcs/aileron-cmd-norm', max(-1.0, min(1.0, aileron)))
        self.fdm.set_property_value('fcs/rudder-cmd-norm', max(-1.0, min(1.0, rudder)))
        # The f16.xml FCS maps throttle cmd 0..1 onto idle..full-AB (pos 0..2): the first
        # half is idle..mil, the second half is the afterburner range. Sandbox stick maps
        # proportionally onto idle..mil; post-combustion at full throttle selects full AB.
        throttle_cmd = 0.5 * max(0.0, min(1.0, thrust_level))
        if physics_parameters.get('post_combustion') and thrust_level >= 0.99:
            throttle_cmd = 1.0
        if wreck_factor < 0.1:
            throttle_cmd = 0.0
        self.fdm.set_property_value('fcs/throttle-cmd-norm', throttle_cmd)

        # terrain elevation (sandbox heightmap) + landing gear state
        terrain_alt = physics_parameters.get('terrain_altitude_m')
        if terrain_alt is not None:
            self._set_if_exists('position/terrain-elevation-asl-ft', terrain_alt * M_TO_FT)
        self._set_if_exists('gear/gear-cmd-norm', 1.0 if physics_parameters.get('gear_down') else 0.0)

        self.fdm.run()

        if DEBUG:
            self._dbg_n = getattr(self, '_dbg_n', 0) + 1
            if self._dbg_n % 30 == 0:
                print('JSBDBG t=%.1f ang=(%.2f,%.2f,%.2f) thr_cmd=%.2f pos=%.2f thrust=%.0f rud=%.2f ail=%.2f elev=%.2f beta=%.1f hdg=%.1f v=%.0f alt=%.0f' % (
                    self.fdm.get_sim_time(),
                    angular.x, angular.y, angular.z,
                    self.fdm.get_property_value('fcs/throttle-cmd-norm'),
                    self.fdm.get_property_value('fcs/throttle-pos-norm'),
                    self.fdm.get_property_value('propulsion/engine/thrust-lbs'),
                    self.fdm.get_property_value('fcs/rudder-cmd-norm'),
                    self.fdm.get_property_value('fcs/aileron-cmd-norm'),
                    self.fdm.get_property_value('fcs/elevator-cmd-norm'),
                    beta, psi,
                    self.fdm.get_property_value('velocities/vc-kts') * KT_TO_MS,
                    self.fdm.get_property_value('position/h-sl-ft') * FT_TO_M))

        # --- FDM state -> Harfang world ---
        pos = self._current_world_position()
        theta = float(self.fdm.get_property_value('attitude/theta-deg'))
        psi = float(self.fdm.get_property_value('attitude/psi-deg'))
        phi = float(self.fdm.get_property_value('attitude/phi-deg'))

        # safety net: deep Euler singularities or solver blowups re-sync from the
        # sandbox's current (pre-update) state instead of propagating garbage.
        state_vals = (pos.x, pos.y, pos.z, theta, psi, phi)
        if any(math.isnan(v) for v in state_vals) or abs(theta) > 89.9 or abs(pos.y) > 30000.0:
            self.sync_from_kinematics(matrix, physics_parameters.get('v_move', hg.Vec3(0, 0, 0)), thrust_level)
            pos = self._current_world_position()
            theta = float(self.fdm.get_property_value('attitude/theta-deg'))
            psi = float(self.fdm.get_property_value('attitude/psi-deg'))
            phi = float(self.fdm.get_property_value('attitude/phi-deg'))
            self._last_rates = (0.0, 0.0, 0.0)

        # rates actually measured across THIS FDM step (post-run vs pre-run), stored
        # for the next frame's damping terms and for the trajectory prediction
        q_new = max(-3.0, min(3.0, math.radians((theta - theta0) / dt)))       # + nose up
        p_new = max(-3.0, min(3.0, math.radians(((phi - phi0 + 540.0) % 360.0 - 180.0) / dt)))  # + roll right
        r_new = max(-3.0, min(3.0, math.radians(((psi - psi0 + 540.0) % 360.0 - 180.0) / dt)))  # + right turn
        self._last_rates = (q_new, p_new, r_new)

        heading = psi % 360.0

        v_move = (pos - self._prev_pos) * (1.0 / dt)
        matrix = self._matrix_from_attitudes(heading, theta, phi, pos)

        self._last_speed_ms = hg.Len(v_move)
        self._last_vy = v_move.y
        self._prev_euler = (theta, psi, phi)
        self._prev_pos = pos

        # cache the kinematic state for the HUD trajectory prediction
        self._pred_state = (pos, self._last_speed_ms, heading, theta, phi,
                            math.degrees(r_new), math.degrees(q_new),
                            (self._last_speed_ms - self._prev_speed_ms) / dt)
        self._prev_speed_ms = self._last_speed_ms

        return matrix, {
            'v_move': v_move,
            'pitch_attitude': theta,
            'heading': heading,
            'roll_attitude': -phi,  # legacy convention: positive = left bank (right wing up)
        }
