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

# --- control conventions (verified empirically, see tools/test_jsbsim_physics.py) ---
# sandbox stick: angular.x > 0 = nose up, angular.y > 0 = yaw left, angular.z > 0 = roll left
# jsbsim f16.xml: elevator-cmd > 0 = nose down, aileron-cmd > 0 = roll right, rudder-cmd > 0 = nose left

# --- SAS gains ---
# f16.xml control conventions (empirical): elevator-cmd > 0 = nose down, aileron-cmd > 0 = roll right,
# rudder-cmd > 0 = nose left. Attitude commands integrate the stick so that a NEUTRAL stick holds the
# current attitude, matching the legacy sandbox where zero angular_level produced no rotation.
PITCH_CMD_RATE = 40.0    # deg/s of pitch command per unit stick
PITCH_CMD_MAX = 70.0     # deg
K_THETA = 0.05           # elevator per deg of pitch error
K_Q = 2.2                # elevator per rad/s pitch rate
K_TRIM = 0.012           # slow integral on pitch error (steady-state attitude hold)
TRIM_MAX = 0.5           # elevator trim integrator clamp
ALPHA_SOFT_DEG = 12.0    # soft AoA limiter: relaxes the pitch cmd above this alpha
ALPHA_LIMIT_DEG = 24.0   # hard AoA protection above this alpha
ALPHA_PROTECT = 0.06     # elevator per degree over the limit
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


def configure(physics_config):
    """Apply the config.json "Physics" section (engine, origin)."""
    global USE_JSBSIM
    if not physics_config:
        return
    USE_JSBSIM = str(physics_config.get('engine', 'jsbsim')).lower() == 'jsbsim'


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
        self._pitch_cmd_deg = 0.0             # SAS attitude commands (integrated from stick)
        self._roll_cmd_deg = 0.0
        self._elev_trim = 0.0                 # slow integrator canceling the untrimmed moment
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
        # note: tank[0] is unindexed in this jsbsim build's property tree
        for name in ('propulsion/tank', 'propulsion/tank[1]', 'propulsion/tank[2]', 'propulsion/tank[3]'):
            try:
                cap = self.fdm.get_property_value(name + '/capacity-lbs')
                self.fdm.set_property_value(name + '/contents-lbs', cap)
            except Exception:
                pass

    # ------------------------------------------------------------------ state sync

    def sync_from_kinematics(self, matrix, v_move, thrust_level=0.8):
        """(Re)initialize the FDM from sandbox state. Called on spawn/reset/teleport and
        whenever the sandbox externally overwrites matrix or velocity."""
        heading, pitch, roll = self._attitudes_from_matrix(matrix)
        pos = hg.GetT(matrix)
        speed = hg.Len(v_move)
        if speed < MIN_AIRSPEED_MS and pos.y > MIN_CONTROLLABLE_ALT_M:
            speed = MIN_AIRSPEED_MS  # airborne with no speed: give a flyable fallback
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
        self._pitch_cmd_deg = pitch
        self._roll_cmd_deg = roll
        self._elev_trim = 0.0
        self._last_speed_ms = speed
        self._last_vy = hg.Len(v_move) * math.sin(math.radians(gamma)) if hg.Len(v_move) > 1.0 else 0.0
        self._prev_pos = self._current_world_position()
        self._ready = True

    def _current_world_position(self):
        lat = float(self.fdm.get_property_value('position/lat-gc-deg'))
        lon = float(self.fdm.get_property_value('position/long-gc-deg'))
        alt = float(self.fdm.get_property_value('position/h-sl-ft')) * FT_TO_M
        return self._world_from_geodetic(lat, lon, alt)

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

        # --- rate estimates from Euler finite differences (fixed dt) ---
        theta = self.fdm.get_property_value('attitude/theta-deg')
        psi = self.fdm.get_property_value('attitude/psi-deg')
        phi = self.fdm.get_property_value('attitude/phi-deg')
        prev_theta, prev_psi, prev_phi = self._prev_euler
        dt = self.FDM_DT
        dpsi = (psi - prev_psi + 540.0) % 360.0 - 180.0
        q_rate = math.radians((theta - prev_theta) / dt)     # + nose up
        p_rate = math.radians(((phi - prev_phi + 540.0) % 360.0 - 180.0) / dt)  # + roll right
        r_rate = math.radians(dpsi / dt)                     # + heading increase (right turn)
        # Euler wrap/singularity can spike the estimates; keep the SAS bounded.
        q_rate = max(-3.0, min(3.0, q_rate))
        p_rate = max(-3.0, min(3.0, p_rate))
        r_rate = max(-3.0, min(3.0, r_rate))

        alpha = self.fdm.get_property_value('aero/alpha-deg')
        beta = self.fdm.get_property_value('aero/beta-deg')

        # --- SAS: integrate stick into attitude commands, then hold them ---
        # elevator/aileron/rudder signs follow the f16.xml conventions stated above;
        # roll command uses -= because sandbox angular.z > 0 means roll LEFT (phi decreases).
        self._pitch_cmd_deg = max(-PITCH_CMD_MAX, min(PITCH_CMD_MAX,
                                     self._pitch_cmd_deg + PITCH_CMD_RATE * angular.x * authority * dt))
        self._roll_cmd_deg = max(-ROLL_CMD_MAX, min(ROLL_CMD_MAX,
                                    self._roll_cmd_deg - (ROLL_CMD_RATE * angular.z
                                                          + YAW_ROLL_RATE * angular.y) * authority * dt))
        # easy-steering behavior (like the legacy engine): with lateral sticks neutral
        # the bank command eases back toward wings-level, preventing locked-in spiral dives.
        if abs(angular.z) < 0.05 and abs(angular.y) < 0.05:
            self._roll_cmd_deg *= max(0.0, 1.0 - dt * LEVEL_DECAY)

        # Flight-path envelope protection (FLCS-style): the commanded attitude is clamped
        # to the current flight-path angle +-15 deg, which bounds alpha so the aircraft
        # can never be commanded into an unsustainable climb / departure. Above
        # ALPHA_SOFT the command relaxes further so speed recovers.
        cmd_eff = self._pitch_cmd_deg
        protected = False
        if self._last_speed_ms > 30.0:
            gamma_deg = math.degrees(math.asin(max(-1.0, min(1.0, self._last_vy / self._last_speed_ms))))
            cmd_eff = max(gamma_deg - 20.0, min(gamma_deg + 15.0, cmd_eff))
            protected = cmd_eff < self._pitch_cmd_deg - 0.01
            cmd_eff -= max(0.0, alpha - ALPHA_SOFT_DEG)
            protected = protected or alpha > ALPHA_SOFT_DEG

        pitch_err = theta - cmd_eff
        if not protected:  # anti-windup: the trim integrator only runs when unclamped
            self._elev_trim = max(-TRIM_MAX, min(TRIM_MAX, self._elev_trim + K_TRIM * pitch_err * dt))
        elevator = K_THETA * pitch_err + K_Q * q_rate + self._elev_trim
        if alpha > ALPHA_LIMIT_DEG:
            elevator += ALPHA_PROTECT * (alpha - ALPHA_LIMIT_DEG)
        elif alpha < -10.0:
            elevator += ALPHA_PROTECT * (-10.0 - alpha)

        aileron = -K_PHI * (phi - self._roll_cmd_deg) - K_P * p_rate  # aileron + = roll right (phi +)

        # expected turn rate for a coordinated bank (deg/s) so the damper doesn't fight it
        speed = max(self.fdm.get_property_value('velocities/vc-kts') * KT_TO_MS, 50.0)
        turn_rate_deg_s = math.degrees(9.81 * math.tan(math.radians(phi)) / speed) if abs(phi) < 80 else 0.0
        rudder = -YAW_FF * angular.y * authority + K_R * math.radians(r_rate - turn_rate_deg_s) + K_BETA * beta
        # (rudder-cmd > 0 = nose left; correcting positive sideslip beta needs nose right = negative cmd)

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

        heading = psi % 360.0

        v_move = (pos - self._prev_pos) * (1.0 / dt)
        matrix = self._matrix_from_attitudes(heading, theta, phi, pos)

        self._last_speed_ms = hg.Len(v_move)
        self._last_vy = v_move.y
        self._prev_euler = (theta, psi, phi)
        self._prev_pos = pos

        return matrix, {
            'v_move': v_move,
            'pitch_attitude': theta,
            'heading': heading,
            'roll_attitude': -phi,  # legacy convention: positive = left bank (right wing up)
        }
