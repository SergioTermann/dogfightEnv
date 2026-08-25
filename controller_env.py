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
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
table = PrettyTable()
import harfang as hg
from random import uniform
from math import radians
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

class controller_env(Env):
    def __init__(self, host='10.134.100.116', port='50888', rendering=True) -> None:
        self.host = host
        self.port = port
        self.rendering = rendering
        self.step_game = 0  #给本局设定结束条件，初定500步
        self.missle_count = 0 #记录导弹的数量，发射的越多罚分越多

        self.last_obs = None
        self.total_step = 0
        try:
            df.get_planes_list()
        except:
            print('Run for the first time')
            df.connect(host, int(port))
            time.sleep(2)
        self.planes = df.get_planes_list()
        self.planeID = self.planes[0]
        self.enemyID = self.planes[1]
        df.disable_log()
        self.planeID = self.planes[0]
        for i in self.planes:
           df.reset_machine(i)
        df.set_plane_thrust(self.planeID, 1)
        df.set_plane_thrust(self.enemyID, 0.5)
        df.set_plane_linear_speed(self.planeID, 300)
        df.set_plane_linear_speed(self.enemyID, 300)
        df.set_client_update_mode(True)
        df.get_targets_list(self.planeID)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)
        #收起起落架
        df.retract_gear(self.planes[0])
        self.action_space = Box(
            low=np.array([
                -1,  # Roll 俯仰角
                -1,  # Pitch 翻滚角
                -1,  # Yaw 偏航角
                 0,  # thrust 油门
                 0,  # fire 发射导弹
            ]),
            high=np.array([
                1,
                1,
                1,
                1,
                1,
            ]),
        )

        self.observation_space = Box(
            low=np.array([  # simple normalized
                -300,  # x / 100
                -300,  # y / 100
                -1,    # z / 50
                -360,  # roll_attitude * 4
                -360,  # pitch_attitude * 4
                 0,    # heading
                 0,    # thrust level 油门
                 0,    # linear speed 空速/1000
                 -1,   # vertical speed 垂直速度/300
                 0,    # horizontal speed 水平速度/500
                -300,  # x / 100 enemy
                -300,  # y / 100 enemy
                -1,    # z / 50  enemy
                 0,    # 是否被锁定
                 0,    # 是否锁定
                 0,    # 距离
                 0,    # 敌军的生命值
                 0,    # 视线角度
            ]),
            high=np.array([
                300,   # x / 100
                300,   # y / 100
                200,   # z / 50
                360,   # roll_attitude * 4
                360,   # pitch_attitude * 4
                360,   # heading
                1,     # thrust level 油门
                1,     # linear speed 空速/1000
                1,     # vertical speed 垂直速度/300
                1,     # horizontal speed 水平速度/500
                300,   # x / 100 enemy
                300,   # y / 100 enemy
                200,   # z /50 enemy
                1,     # 是否被锁定
                1,     # 是否锁定
                100000,# 距离
                1,     # 敌军的生命值
                90,    # 视线角度
            ])
        )
        self.replay_buffer = ReplayBuffer(100000, self.observation_space, self.action_space)
    def getDistance(self):
        return ((df.get_plane_state(self.planeID)['position'][0] - df.get_plane_state(self.enemyID)['position'][0]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][1] - df.get_plane_state(self.enemyID)['position'][1]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][2] - df.get_plane_state(self.enemyID)['position'][2]) ** 2) ** .5


    def step(self):
        self.step_game += 1

        df.update_scene()
        reward = self.reward()
        terminate = True if self.terminate() else False
        plane_state = df.get_plane_state(self.planeID)
        plan_loc = plane_state['position']
        enemy_loc = df.get_plane_state(self.enemyID)['position']

        ob = [  # normalized
            plan_loc[0] / 1000,
            plan_loc[2] / 1000,
            plan_loc[1] / 1000,
            plane_state['roll_attitude'] / 90,
            plane_state['pitch_attitude'] / 90,
            plane_state['heading'] / 360,
            plane_state['thrust_level'],
            plane_state['linear_speed']/1000,
            plane_state['vertical_speed']/300,
            plane_state['horizontal_speed']/500,
            enemy_loc[0] / 1000,
            enemy_loc[2] / 1000,
            enemy_loc[1] / 1000,
            df.get_plane_state(self.enemyID)['target_locked'],
            df.get_plane_state(self.planeID)['target_locked'],
            self.getDistance()/1000,
            self.getEnemyHP(),
            self.angle_attacking(plane_state['heading'] , plane_state['pitch_attitude'], plan_loc, enemy_loc) / 180
        ]

        df.update_scene()
        return ob, reward, terminate, {}

    def terminate(self):
        return 0

    def reset(self):
        self.step_game = 0  #给本局设定结束条件，初定500步
        self.missle_count = 0 #记录导弹的数量，发射的越多罚分越多
        for i in self.planes:
            df.reset_machine(i)
        df.set_plane_thrust(self.planeID, 1)
        df.set_plane_thrust(self.enemyID, 0.5)
        df.set_client_update_mode(True)

        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)
        # 收起起落架
        plane_state = df.get_plane_state(self.planeID)
        plan_loc = plane_state['position']
        enemy_loc = df.get_plane_state(self.enemyID)['position']
        df.get_targets_list(self.planeID)

        ob = [  # normalized
            plan_loc[0] / 1000,
            plan_loc[2] / 1000,
            plan_loc[1] / 1000,
            plane_state['roll_attitude'] / 90,
            plane_state['pitch_attitude'] / 90,
            plane_state['heading'] / 360,
            plane_state['thrust_level'],
            plane_state['linear_speed']/1000,
            plane_state['vertical_speed']/300,
            plane_state['horizontal_speed']/500,
            enemy_loc[0] / 1000,
            enemy_loc[2] / 1000,
            enemy_loc[1] / 1000,
            df.get_plane_state(self.enemyID)['target_locked'],
            df.get_plane_state(self.planeID)['target_locked'],
            self.getDistance()/1000,
            self.getEnemyHP(),
            self.angle_attacking(plane_state['heading'] , plane_state['pitch_attitude'], plan_loc, enemy_loc) / 180
        ]
        self.last_obs = ob
        self.missle_count = 0
        self.step_game = 0
        range = hg.Vec3(0, 0, 0)
        center = hg.Vec3(1000, 2000, 3500)
        y_orientations_range = hg.Vec2(-45, 45)
        df.reset_machine_matrix(self.enemyID,
                                uniform(center.x - range.x / 2, center.x + range.x / 2),
                                uniform(center.y - range.y / 2, center.y + range.y / 2),
                                uniform(center.z - range.z / 2, center.z + range.z / 2),
                                0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        center = hg.Vec3(1000, 2000, 1500)
        range_plane = hg.Vec3(0, 0, 0)
        y_orientations_range = hg.Vec2(-0, 0)
        df.reset_machine_matrix(self.planeID,
                                uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
                                uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
                                uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
                                0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        df.set_plane_linear_speed(self.planeID, 300)
        df.set_plane_linear_speed(self.enemyID, 300)
        return ob

    def reward(self):
        if self.terminate() == 1:
            reward = 50
        if df.get_plane_state(self.enemyID)["health_level"] <= 0.9:
            reward = 50
        elif self.terminate() == -1:
            reward = -50
        else:
            reward = -0.01
        if df.get_plane_state(self.enemyID)['target_locked']:
            reward -= 10
        reward += (3-self.missle_count)*0.01
        if df.get_plane_state(self.planeID)['target_locked']:
            reward += 0.05
        return reward

    def angle_attacking(self, heading, pitch, plane_loc, enemy_loc):
        x1 = enemy_loc[2] - plane_loc[2]
        y1 = enemy_loc[0] - plane_loc[0]
        z1 = enemy_loc[1] - plane_loc[1]
        z = np.sin(pitch/180*np.pi)
        y = np.cos(pitch/180*np.pi)*np.sin(heading/180*np.pi)
        x = np.cos(pitch/180*np.pi)*np.cos(heading/180*np.pi)
        angle = np.arccos((x1*x + y1*y + z1*z)/np.sqrt(x1**2+y1**2+z1**2))
        return angle/np.pi*180

    def getHP(self):
        return df.get_health(self.planeID)['health_level']

    def getEnemyHP(self):
        return df.get_health(self.enemyID)['health_level']
