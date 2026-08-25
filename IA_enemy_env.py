import random
import re
import sys
import time
import warnings

import gym
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete

sys.path.append('./src/')
sys.path.append('./src/environments/dogfightEnv/')
sys.path.append('./src/environments/dogfightEnv/dogfight_sandbox_hg2/network_client_example/')
# sys.path.append('gym.envs.dogfightEnv.dogfight_sandbox_hg2.network_client_example/')
sys.path.append('gym.envs.dogfightEnv.dogfight_sandbox_hg2')

try:
    from .dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("Gym in IA_enemy_env")
    time.sleep(1)
except:
    from dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("DBRL")
    time.sleep(1)

class IA_enemy_env(Env):

    def __init__(self, host='10.134.100.34', port='50888', rendering=True) -> None:
        self.host = host
        self.port = port
        self.nof = 0
        self.rendering = rendering
        self.step_game = 0  #给本局设定结束条件，初定500步
        self.missle_count = 0 #记录导弹的数量，发射的越多罚分越多
        self.enemy_set_mark = False
        self.target_lock_count = 0
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

        self.planeID = planes[0]

        for i in planes:
           df.reset_machine(i)

        df.set_plane_thrust(self.planeID, 1)
        df.set_plane_thrust(self.enemyID, 1)

        df.set_client_update_mode(True)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)

        df.retract_gear(planes[0])

        #activate IA
        df.activate_IA(self.enemyID)

        missles = df.get_machine_missiles_list(self.planeID)

        self.action_space = Box(
            low=np.array([
                -1,  # Roll 俯仰角
                -1,  # Pitch 翻滚角
                -1,  # Yaw 偏航角
                -1,  # flaps 襟翼
                -1,  # break 刹车
                 0,  # throattle 油门
                 0,  # fire 发射导弹
                 # 0,  # target device 更换瞄准目标
            ]),
            high=np.array([
                1,
                1,
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
                 0,    # break level 刹车
                 0,    # flaps 襟翼
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
                1,     # break level 刹车
                1,     # flaps 襟翼
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

    def getHP(self):
        return df.get_health(self.planeID)['health_level']

    def getEnemyHP(self):
        # print(df.get_health(self.enemyID)['health_level'])
        return df.get_health(self.enemyID)['health_level']

    def render(self, id=0):
        df.set_renderless_mode(False)

    def step(self, action):
        self.step_game += 1
        self.sendAction(action)

        self.nof += 1
        reward = self.reward()
        terminate = True if self.terminate() else False
        plane_state = df.get_plane_state(self.planeID)
        plan_loc = plane_state['position']
        enemy_loc = df.get_plane_state(self.enemyID)['position']

        ob = [  # normalized
            plan_loc[0] / 100,
            plan_loc[2] / 100,
            (plan_loc[1] - 1000) / 100,
            plane_state['roll_attitude'] / 90,
            plane_state['pitch_attitude'] / 90,
            plane_state['heading'] / 90,
            plane_state['thrust_level'],
            plane_state['brake_level'],
            plane_state['flaps_level'],
            plane_state['linear_speed']/1000,
            plane_state['vertical_speed']/300,
            plane_state['horizontal_speed']/500,
            enemy_loc[0] / 100,
            enemy_loc[2] / 100,
            (enemy_loc[1] - 1000) / 100,
            df.get_plane_state(self.enemyID)['target_locked'],
            df.get_plane_state(self.planeID)['target_locked'],
            self.getDistance()/1000,
            self.getEnemyHP(),
            self.angle_attacking(plane_state['heading'] , plane_state['pitch_attitude'], plan_loc, enemy_loc) / 180
        ]
        plane_action = df.get_plane_action(self.planeID)

        print(df.get_plane_action(self.planeID))
        if self.step_game % 200 == 0:
            print(ob)
        df.update_scene()

        return ob, reward, terminate, {}

    def reward(self):
        if self.terminate() == 1:
            reward = 50
        if df.get_plane_state(self.enemyID)["health_level"] <= 0.9:
            reward = 50
        elif self.terminate() == 2:
            reward = -50
        else:
            reward = 0
        if df.get_plane_state(self.enemyID)['target_locked']:
            reward -= 0.05
        reward += (3-self.missle_count)*0.01
        if df.get_plane_state(self.planeID)['target_locked']:
            reward += 0.05

        return reward
    
    def terminate(self):

        if self.step_game >= 2048:
            return 3
        # 判定HP首先小于80的人战败
        if self.getHP() <= .8:
            return 2
        elif df.get_plane_state(self.enemyID)["health_level"] <= 0.8:
            return 1
        else:
            return 0

    def getDistance(self):
        return ((df.get_plane_state(self.planeID)['position'][0] - df.get_plane_state(self.enemyID)['position'][0]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][1] - df.get_plane_state(self.enemyID)['position'][1]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][2] - df.get_plane_state(self.enemyID)['position'][2]) ** 2) ** .5

    def reset(self):
        self.__init__(self.host, self.port, self.rendering)

        plane_state = df.get_plane_state(self.planeID)
        plan_loc = plane_state['position']
        enemy_loc = df.get_plane_state(self.enemyID)['position']
        ob = [  # normalized
            plan_loc[0] / 100,
            plan_loc[2] / 100,
            (plan_loc[1] - 1000) / 100,
            plane_state['roll_attitude'] / 90,
            plane_state['pitch_attitude'] / 90,
            plane_state['heading'] / 90,
            plane_state['thrust_level'],
            plane_state['brake_level'],
            plane_state['flaps_level'],
            plane_state['linear_speed']/1000,
            plane_state['vertical_speed']/300,
            plane_state['horizontal_speed']/500,
            enemy_loc[0] / 100,
            enemy_loc[2] / 100,
            (enemy_loc[1] - 1000) / 100,
            df.get_plane_state(self.enemyID)['target_locked'],
            df.get_plane_state(self.planeID)['target_locked'],
            self.getDistance()/1000,
            self.getEnemyHP(),
            self.angle_attacking(plane_state['heading'] , plane_state['pitch_attitude'], plan_loc, enemy_loc) / 180
        ]
        self.missle_count = 0
        self.enemy_set_mark = False
        self.step_game = 0
        return ob

    def sendAction(self, action, actionType=None):
        # df.set_plane_yaw(self.planeID, float(action[0]))
        # df.set_plane_pitch(self.planeID, float(action[1]))
        # df.set_plane_roll(self.planeID, float(action[2]))
        # df.set_plane_thrust(self.planeID, float(action[3]))

        # df.set_plane_yaw(self.planeID, float(action[0]))
        df.set_plane_pitch(self.planeID, float(action[0]))
        df.set_plane_roll(self.planeID, float(action[1]))
        df.set_plane_thrust(self.planeID, float(action[2]))
        if action[3] > 0.8 and self.missle_count < 3:
            self.missle_count += 1
            df.fire_missile(self.planeID, self.missle_count)

    def angle_attacking(self, heading, pitch, plane_loc, enemy_loc):
        x1 = enemy_loc[2] - plane_loc[2]
        y1 = enemy_loc[0] - plane_loc[0]
        z1 = enemy_loc[1] - plane_loc[1]
        # print('enemy-plane vector is', x1, y1, z1)
        # print('pitch is ', pitch)
        z = np.sin(pitch/180*np.pi)
        y = np.cos(pitch/180*np.pi)*np.sin(heading/180*np.pi)
        x = np.cos(pitch/180*np.pi)*np.cos(heading/180*np.pi)
        # print('xyz angle is', x, y, z)
        angle = np.arccos((x1*x + y1*y + z1*z)/np.sqrt(x1**2+y1**2+z1**2))
        return angle/np.pi*180