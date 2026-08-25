import math
import random
import re
import sys
import time
import warnings

import gym
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from prettytable import PrettyTable
import harfang as hg
from random import uniform
from math import radians
table = PrettyTable()
CONSTANTS_RADIUS_OF_EARTH = 6371000.0  # meters, used by XYtoGPS (Tacview logging)
sys.path.append('./src/')
sys.path.append('./src/environments/dogfightEnv/')
sys.path.append('./src/environments/dogfightEnv/dogfight_sandbox_hg2/network_client_example/')
sys.path.append('gym.envs.dogfightEnv.dogfight_sandbox_hg2')
try:
    from .dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("Gym in oneVSone")
    time.sleep(1)

except:
    from dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("DBRL")
    time.sleep(1)


def XYtoGPS(x, y, ref_lat=24.8976763, ref_lon=160.123456):
    x_rad = float(x) / CONSTANTS_RADIUS_OF_EARTH
    y_rad = float(y) / CONSTANTS_RADIUS_OF_EARTH
    c = math.sqrt(x_rad * x_rad + y_rad * y_rad)

    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    ref_sin_lat = math.sin(ref_lat_rad)
    ref_cos_lat = math.cos(ref_lat_rad)

    if abs(c) > 0:
        sin_c = math.sin(c)
        cos_c = math.cos(c)

        lat_rad = math.asin(cos_c * ref_sin_lat + (x_rad * sin_c * ref_cos_lat) / c)
        lon_rad = (ref_lon_rad + math.atan2(y_rad * sin_c, c * ref_cos_lat * cos_c - x_rad * ref_sin_lat * sin_c))

        lat = math.degrees(lat_rad)
        lon = math.degrees(lon_rad)

    else:
        lat = math.degrees(ref_lat)
        lon = math.degrees(ref_lon)

    return lat, lon


class oneVSoneEnv(Env):

    def __init__(self, host='10.134.100.34', port='50888', rendering=True) -> None:
        self.host = host
        self.port = port
        self.nof = 0
        self.rendering = rendering
        self.step_game = 0  # 给本局设定结束条件，初定500步
        self.missle_count = 0  # 记录导弹的数量，发射的越多罚分越多
        self.enemy_set_mark = False
        self.target_lock_count = 0
        self.total_step = 0
        self.name = 'oneVSone'
        self.facing = 0
        self.aming = 0
        self.ally = []
        self.enemy = []
        self.ally_missile = []
        try:
            df.get_planes_list()
        except:
            print('Run for the first time')
            df.connect(host, int(port))
            time.sleep(2)
        planes = df.get_planes_list()
        self.planeID = planes[0]
        self.enemyID = planes[1]
        df.disable_log()

        # 设定本方战机
        self.planeID = planes[0]

        # 为所有战机初始化 从这里开始显示各架飞机
        for i in planes:
            df.reset_machine(i)
        df.get_targets_list(self.planeID)
        # 两架飞机先飞着
        df.set_plane_thrust(self.planeID, 1)
        df.set_plane_thrust(self.enemyID, 1)
        # 设置成用户模式
        df.set_client_update_mode(True)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)
        # 收起起落架
        df.retract_gear(planes[0])
        missles = df.get_machine_missiles_list(self.planeID)
        df.set_plane_linear_speed(self.planeID, 300)
        df.set_plane_linear_speed(self.enemyID, 300)
        self.action_space = Box(
            low=np.array([
                -1,  # Roll 俯仰角
                -1,  # Pitch 翻滚角
                -1,  # Yaw 偏航角
                # -1,  # flaps 襟翼
                # -1,  # break 刹车
                 0,  # thrust 油门
                 0,  # fire 发射导弹
                 # 0,  # target device 更换瞄准目标
            ]),
            high=np.array([
                1,
                1,
                1,
                # 1,
                # 1,
                1,
                1,
            ]),
        )

        self.observation_space = Box(
            low=np.array([  # simple normalized
                -300,  # x / 1000
                -300,  # y / 1000
                -1,    # z / 1000
                -360,  # roll_attitude * 4
                -360,  # pitch_attitude * 4
                0,  # heading
                0,  # thrust level 油门
                0,  # linear speed 空速/1000
                -1,  # vertical speed 垂直速度/300
                0,  # horizontal speed 水平速度/500
                0,  # 是否锁定

                -300,  # x / 100 enemy
                -300,  # y / 100 enemy
                -1,    # z / 50  enemy
                -360,  # roll_attitude * 4
                -360,  # pitch_attitude * 4
                0,  # heading
                0,  # thrust level 油门
                0,  # linear speed 空速/1000
                -1,  # vertical speed 垂直速度/300
                0,  # horizontal speed 水平速度/500
                0,  # 是否锁定

                0,  # missile count 导弹发射计数/3
                0,    # 距离
                -180,    # 罗盘角度
                0,  # 视线角度
            ]),
            high=np.array([
                300,  # x / 1000
                300,  # y / 1000
                300,  # z / 1000
                360,  # roll_attitude * 4
                360,  # pitch_attitude * 4
                360,  # heading
                1,  # thrust level 油门
                100,  # linear speed 空速/1000
                100,  # vertical speed 垂直速度/300
                100,  # horizontal speed 水平速度/500
                1,  # 是否锁定

                300,  # x / 100 enemy
                300,  # y / 100 enemy
                300,  # z / 50  enemy
                360,  # roll_attitude * 4
                360,  # pitch_attitude * 4
                360,  # heading
                1,  # thrust level 油门
                100,  # linear speed 空速/1000
                100,  # vertical speed 垂直速度/300
                100,  # horizontal speed 水平速度/500
                1,  # 是否锁定

                1,  # missile count 导弹发射计数/3
                1000,  # 距离
                180,  # 罗盘角度
                180,  # 视线角度
            ])
        )

    def render(self, id=0):
        df.set_renderless_mode(False)

    def step(self, action):
        file = open('trained_epoch_0.txt', 'a')  # was a hardcoded foreign absolute path
        file.write('#{}'.format(self.step_game/100)+'\n')

        self.step_game += 1
        self.total_step += 1
        self.sendAction(action)
        
        if self.step_game % 200 == 0:
            df.set_plane_pitch(self.enemyID, np.random.random()-0.5)
            df.set_plane_roll(self.enemyID, np.random.random()-0.5)
        df.update_scene()
        while True:
            flag = df.get_finish_flag()
            if flag:
                break

        for i, aircraft in enumerate(self.ally):
            if aircraft == '':
                continue
            else:
                state = df.get_plane_state(aircraft)

                loc = state['position']
                lat, lon = XYtoGPS(loc[0], loc[2])
                euler_angle = state['Euler_angles']
                roll = euler_angle[2] / np.pi * 180
                pitch = euler_angle[0] / np.pi * 180
                yaw = ((360 - euler_angle[1] / np.pi * 180) + 90) % 360
                missile = df.get_missiles_list()
                missile_state = df.get_missile_state(missile[0])
                missile_state1 = df.get_missile_state(missile[1])
                if state['health_level']:
                    file.write("{},T={}|{}|{}|{}|{}|{},Type=Air+FixedWing,Coalition=Enemies,Color=Red,Name=F-16,Mach=0.800,"
                               "ShortName=F-16  0  3.00,RadarMode=1,RadarRange=2000,RadarHorizontalBeamwidth=10,"
                               "RadarVerticalBeamwidth=10".format(i+1, lon, lat, loc[1], roll, -pitch, yaw)+"\n")
                else:
                    file.write("{},T=||,Visible=0".format(i+1) + "\n")

                    self.ally[self.ally.index(aircraft)] = ''
        for im, missile_am in enumerate(self.ally_missile):
            if missile_am == '':
                continue
            else:
                if df.get_missile_state(missile_am)['position'][0]:
                    state_am = df.get_missile_state(missile_am)
                    euler_angle_am = state_am['Euler_angles']
                    loc_am = state_am['position']
                    lat_am, lon_am = XYtoGPS(loc_am[0], loc_am[2])
                    roll_am = euler_angle_am[2] / np.pi * 180
                    pitch_am = euler_angle_am[0] / np.pi * 180
                    yaw_am = ((360 - euler_angle_am[1] / np.pi * 180) + 90) % 360
                    if state_am['wreck']: #state_am['crashed'] or state_am['destroyed'] or
                        file.write("{},T=||,Visible=0".format(im+100) + '\n')
                        self.explosion_am.append((im+2) * 1000)
                        self.explosion_am_index.append(10)
                        file.write(
                            "{},T={}|{}|{}|||,Color=Blue,Type=Misc+Explosion,Radius=250".format((im+2) * 1000, lon_am,
                                                                                                lat_am, loc_am[1]) + "\n")
                        self.ally_missile[self.ally_missile.index(missile_am)] = ''
                    else:
                        file.write(
                            "{},T={}|{}|{}|{}|{}|{},Type=Medium+Weapon+Missile,Coalition=Allies,Color=Red,Name=Mica,Mach=1,"
                            .format(im+100, lon_am, lat_am, loc_am[1], roll_am, -pitch_am, yaw_am) + "\n")

        for j, aircraft_enemy in enumerate(self.enemy):
            if aircraft_enemy == '':
                continue
            else:
                show_count_enemy = 0
                state_enemy = df.get_plane_state(aircraft_enemy)

                loc_enemy = state_enemy['position']
                lat_enemy, lon_enemy = XYtoGPS(loc_enemy[0], loc_enemy[2])
                euler_angle_enemy = state_enemy['Euler_angles']
                roll_enemy = euler_angle_enemy[2] / np.pi * 180
                pitch_enemy = euler_angle_enemy[0] / np.pi * 180
                yaw_enemy = ((360 - euler_angle_enemy[1] / np.pi * 180) + 90) % 360
                if state_enemy['health_level']:
                    file.write("{},T={}|{}|{}|{}|{}|{},Type=Air+FixedWing,Coalition=Enemies,Color=Blue,Name=F-16,Mach=0.800,"
                               "ShortName=F-16  0  3.00,RadarMode=1,RadarRange=2000,RadarHorizontalBeamwidth=10,"
                               "RadarVerticalBeamwidth=10".format(j+9, lon_enemy, lat_enemy, loc_enemy[1], roll_enemy,
                                                                  -pitch_enemy, yaw_enemy)+"\n")
                else:
                    file.write("{},T=||,Visible=0".format(j+9) + "\n")
                    self.enemy[self.enemy.index(aircraft_enemy)] = ''

        new_plane_state = df.get_plane_state(self.planeID)
        new_plan_loc = new_plane_state['position']
        new_enemy_loc = df.get_plane_state(self.enemyID)['position']
        new_enemy_state = df.get_plane_state(self.enemyID)

        terminate_value = self.terminate(new_plane_state['health_level'], new_enemy_state['health_level'])
        terminate = True if terminate_value else False
        reward = self.reward(new_plane_state['linear_speed'], terminate_value, new_plane_state['health_level'], new_enemy_state['health_level'], new_plane_state['target_locked'], new_enemy_state['target_locked'])

        Euler_plane = new_plane_state['Euler_angles']
        plane_roll = Euler_plane[2] / np.pi
        plane_pitch = Euler_plane[0] / np.pi
        plane_yaw = Euler_plane[1] / np.pi
        Euler_enemy = new_enemy_state['Euler_angles']
        enemy_roll = Euler_enemy[2] / np.pi
        enemy_pitch = Euler_enemy[0] / np.pi
        enemy_yaw = Euler_enemy[1] / np.pi
        self.facing = self.facing_angle(plane_roll, new_plane_state['heading'], new_plan_loc[0], new_plan_loc[2], new_enemy_loc[0],
                                        new_enemy_loc[2])
        self.aming = self.angle_attacking(new_plane_state['heading'] , new_plane_state['pitch_attitude'], new_plan_loc, new_enemy_loc) / 180
        new_ob = [  # normalized
            new_plan_loc[0] / 10000,
            new_plan_loc[2] / 10000,
            new_plan_loc[1] / 10000,
            plane_roll,
            plane_pitch,
            plane_yaw,
            new_plane_state['thrust_level'],
            new_plane_state['linear_speed'] / 1000,
            new_plane_state['vertical_speed'] / 300,
            new_plane_state['horizontal_speed'] / 500,
            int(new_plane_state['target_locked']),

            new_enemy_loc[0] / 10000,
            new_enemy_loc[2] / 10000,
            new_enemy_loc[1] / 10000,
            enemy_roll,
            enemy_pitch,
            enemy_yaw,
            new_enemy_state['thrust_level'],
            new_enemy_state['linear_speed'] / 1000,
            new_enemy_state['vertical_speed'] / 300,
            new_enemy_state['horizontal_speed'] / 500,
            int(new_enemy_state['target_locked']),

            self.missle_count / 3,
            self.getDistance() / 10000,
            self.facing/180,
            self.aming/90,
        ]
        # roll_true = new_plane_state['Euler_angles'][2]*180/np.pi
        # pitch_true = new_plane_state['Euler_angles'][0]*180/np.pi
        # yaw_true = new_plane_state['Euler_angles'][1]*180/np.pi
        return new_ob, reward, terminate, {}

    def reward(self, speed, terminate, health, enemy_health, lock, enemy_lock):
        if terminate == 1:
            reward = 100
        elif terminate == -1:
            reward = -50
        elif terminate == 2:
            reward = -20
        else:
            reward = 0
        if enemy_lock:
            reward -= 10
        if lock:
            reward += 0.2
        reward += (3-self.missle_count)*0.01
        if self.facing < 30 or self.facing > 330:
            reward += 0.02
        elif abs(self.facing) < 60 or self.facing > 300:
            reward += 0.01
        elif abs(self.facing) > 90 or self.facing < 270:
            reward -= 0.02
        if self.aming < 30:
            reward += 0.01
        if speed < 83:
            reward -= 0.05
        return reward

    def terminate(self, health, enemy_health):

        if self.step_game >= 2048:
            return 2
        if health <= .8:
            return -1
        elif enemy_health <= 0.9:
            return 1
        else:
            return 0

    def getDistance(self):
        return ((df.get_plane_state(self.planeID)['position'][0] - df.get_plane_state(self.enemyID)['position'][0]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][1] - df.get_plane_state(self.enemyID)['position'][1]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][2] - df.get_plane_state(self.enemyID)['position'][2]) ** 2) ** .5

    def reset(self):
        open('trained_epoch_0.txt', "w").close()
        file = open('trained_epoch_0.txt', 'a')
        with open('trained_epoch_0.txt', "w") as file:
            file.write(
                'FileType=text/acmi/tacview' + '\n' + 'FileVersion=2.1' + '\n' + '0,ReferenceTime=2022-10-01T00:00:00Z' + '\n'
                + '0,Title = test simple aircraft' + '\n' +
                '1000000,T=160.123456|24.8976763|0, Type=Ground+Static+Building, Name=Competition, EngagementRange=30000'
                + '\n')
        planes = df.get_planes_list()
        self.step_game = 0  # 给本局设定结束条件，初定500步
        self.missle_count = 0  # 记录导弹的数量，发射的越多罚分越多
        for i in planes:
            df.reset_machine(i)
        self.ally = planes[:1]
        self.enemy = planes[1:]
        self.ally_missile = []  # was `missile[:12]` with missile undefined (crashed reset)
        df.set_plane_thrust(self.planeID, 1)
        df.set_plane_thrust(self.enemyID, 1)
        df.set_client_update_mode(True)
        if self.step_game % 200 == 0:
            df.set_plane_pitch(self.enemyID, np.random.random()-0.5)
            df.set_plane_roll(self.enemyID, np.random.random()-0.5)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)
        # 收起起落架
        df.retract_gear(planes[0])
        new_plane_state = df.get_plane_state(self.planeID)
        new_plan_loc = new_plane_state['position']
        new_enemy_loc = df.get_plane_state(self.enemyID)['position']
        new_enemy_state = df.get_plane_state(self.enemyID)

        df.get_targets_list(self.planeID)
        Euler_plane = new_plane_state['Euler_angles']
        plane_roll = Euler_plane[2] /np.pi
        plane_pitch = Euler_plane[0] /np.pi
        plane_yaw = Euler_plane[1] /np.pi
        Euler_enemy = new_enemy_state['Euler_angles']
        enemy_roll = Euler_enemy[2] / np.pi
        enemy_pitch = Euler_enemy[0] / np.pi
        enemy_yaw = Euler_enemy[1] / np.pi
        self.facing = self.facing_angle(plane_roll, new_plane_state['heading'], new_plan_loc[0], new_plan_loc[2], new_enemy_loc[0],
                                        new_enemy_loc[2])
        self.aming = self.angle_attacking(new_plane_state['heading'] , new_plane_state['pitch_attitude'], new_plan_loc, new_enemy_loc) / 180
        new_ob = [  # normalized
            new_plan_loc[0] / 10000,
            new_plan_loc[2] / 10000,
            new_plan_loc[1] / 10000,
            plane_roll,
            plane_pitch,
            plane_yaw,
            new_plane_state['thrust_level'],
            new_plane_state['linear_speed'] / 1000,
            new_plane_state['vertical_speed'] / 300,
            new_plane_state['horizontal_speed'] / 500,
            int(new_plane_state['target_locked']),

            new_enemy_loc[0] / 10000,
            new_enemy_loc[2] / 10000,
            new_enemy_loc[1] / 10000,
            enemy_roll,
            enemy_pitch,
            enemy_yaw,
            new_enemy_state['thrust_level'],
            new_enemy_state['linear_speed'] / 1000,
            new_enemy_state['vertical_speed'] / 300,
            new_enemy_state['horizontal_speed'] / 500,
            int(new_enemy_state['target_locked']),

            self.missle_count / 3,
            self.getDistance() / 10000,
            self.facing/360,
            self.aming/90,
        ]

        self.last_obs = new_ob
        self.missle_count = 0
        self.step_game = 0
        circle_range = 60 / 180 * np.pi
        
        range = hg.Vec3(500, 500, 500)
        center = hg.Vec3(1000, 2000, 3000)
        y_orientations_range = hg.Vec2(-circle_range, circle_range)
        df.reset_machine_matrix(self.enemyID,
                                uniform(center.x - range.x / 2, center.x + range.x / 2),
                                uniform(center.y - range.y / 2, center.y + range.y / 2),
                                uniform(center.z - range.z / 2, center.z + range.z / 2),
                                0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)

        range_plane = hg.Vec3(500, 500, 500)
        center = hg.Vec3(1000, 2000, 1500)
        y_orientations_range = hg.Vec2(-circle_range, circle_range)
        df.reset_machine_matrix(self.planeID,
                                uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
                                uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
                                uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
                                0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)

        df.set_plane_linear_speed(self.planeID, 85)
        df.set_plane_linear_speed(self.enemyID, 85)
        df.update_scene()
        while True:
            flag = df.get_finish_flag()
            if flag:
                break
        return new_ob

    def sendAction(self, action, actionType=None):
        df.set_plane_roll(self.planeID, float(action[0]))
        df.set_plane_pitch(self.planeID, float(action[1]))
        df.set_plane_yaw(self.planeID, float(action[2]))
        df.set_plane_thrust(self.planeID, float(action[3]))
        if action[4] > 0.8 and self.missle_count < 3:
            self.missle_count += 1
            df.fire_missile(self.planeID, self.missle_count)

    def angle_attacking(self, heading, pitch, plane_loc, enemy_loc):
        x1 = enemy_loc[2] - plane_loc[2]
        y1 = enemy_loc[0] - plane_loc[0]
        z1 = enemy_loc[1] - plane_loc[1]
        z = np.sin(pitch/180*np.pi)
        y = np.cos(pitch/180*np.pi)*np.sin(heading/180*np.pi)
        x = np.cos(pitch/180*np.pi)*np.cos(heading/180*np.pi)
        angle = np.arccos((x1*x + y1*y + z1*z)/np.sqrt(x1**2+y1**2+z1**2))
        return angle/np.pi * 180

    def facing_angle(self, roll, heading, x, y, x_e, y_e):
        x1 = x_e - x
        y1 = y_e - y
        v1 = np.array([x1, y1])
        east = np.array([0, 1])
        cos_heading = np.dot(east, v1)/(np.linalg.norm(east) * np.linalg.norm(v1))
        angle_e = np.arccos(cos_heading) * 180 / np.pi
        if x_e < x:
            result =  -( heading - (360 - angle_e))
        else:
            result = 360 - heading + angle_e
        if result > 180:
            result = -(360 - result)
        if roll*180 > 90 or roll*180 < -90:
            result =- result
        return result