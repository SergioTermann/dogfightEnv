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
    print("Gym in twoVStwo")
    time.sleep(1)
except:
    from dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("DBRL")
    time.sleep(1)

class twoVStwo(Env):

    def __init__(self, host='10.134.100.34', port='50888', rendering=True) -> None:
        self.host = host
        self.port = port
        self.nof = 0
        self.rendering = rendering
        self.step_game = 0  #给本局设定结束条件，初定500步
        self.missle_count = 0 #记录导弹的数量，发射的越多罚分越多
        self.enemy_set_mark = False
        try:
            df.get_planes_list()
        except:
            print('Run for the first time')
            df.connect(host, int(port))
            time.sleep(2)
        planes = df.get_planes_list()
        self.planeID = planes[0]
        self.planeID2 = planes[1]
        self.enemyID = planes[2]
        self.enemyID2 = planes[3]

        print('plane is:', planes)
        df.disable_log()

        #为所有战机初始化 从这里开始显示各架飞机
        for i in planes:
            df.reset_machine(i)
            df.retract_gear(i)

        #两架飞机先飞着
        df.set_plane_thrust(self.planeID, 1)
        df.set_plane_thrust(self.planeID2, 1)
        df.set_plane_thrust(self.enemyID, 0.4)
        df.set_plane_thrust(self.enemyID2, 0.4)

        #设置成用户模式
        df.set_client_update_mode(True)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)

        # df.activate_IA(self.planeID)#在这个版本中加入了
        self.action_space = Box(
            low=np.array([
                -1,  # Yaw 偏航角
                -1, # Pitch 翻滚角
                -1,  # Roll 俯仰角
                0,   # throattle 油门
                0,   # fire 发射导弹
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
                0,     # heading
                -360,  # pitch_attitude * 4
                -360,  # roll_attitude * 4
                -300,  # x / 100 enemy
                -300,  # y / 100 enemy
                -1,    # z / 50  enemy
                0, # 是否锁定
                0,
                0,
                0,
                0,
            ]),
            high=np.array([
                300, # x / 100
                300, # y / 100
                200, # z / 50
                360, # heading
                360, # pitch_attitude * 4
                360, # roll_attitude * 4
                300, # x / 100 enemy
                300, # y / 100 enemy
                200, # z /50 enemy
                1, # 是否被锁定
                1, #是否锁定
                100000,#距离
                1, #敌军的生命值
                90, #视线中距离
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
        # if self.step_game > 200 and self.enemy_set_mark==False:
        #     df.set_plane_pitch(self.enemyID, -0.3)
        #     if self.step_game > 300:
        #         df.set_plane_pitch(self.enemyID, 0)
        #         self.enemy_set_mark = True
        # if self.step_game % 200 == 0:
        #     # df.set_plane_pitch(self.enemyID, 2*np.random.random()-1)
        #     df.set_plane_pitch(self.enemyID, np.random.random()-0.5)
        #     # df.set_plane_roll(self.enemyID, 2*np.random.random()-1)
        #     df.set_plane_roll(self.enemyID, np.random.random()-0.5)
        self.nof += 1
        # print('step once')
        reward = self.reward()
        terminate = True if self.terminate() else False
        heading = df.get_plane_state(self.planeID)['heading']
        pitch = df.get_plane_state(self.planeID)['pitch_attitude']
        plan_loc = df.get_plane_state(self.planeID)['position']
        enemy_loc = df.get_plane_state(self.enemyID)['position']
        ob = [  # normalized
            plan_loc[0] / 100,
            plan_loc[2] / 100,
            (plan_loc[1] - 1000) / 100,
            heading / 90,
            pitch / 90,
            df.get_plane_state(self.planeID)['roll_attitude'] / 90,
            enemy_loc[0] / 100,
            enemy_loc[2] / 100,
            (enemy_loc[1] - 1000) / 100,
            df.get_plane_state(self.enemyID)['target_locked'],
            df.get_plane_state(self.enemyID)['target_locked'],
            self.getDistance()/1000,
            self.getEnemyHP(),
            self.angle_attacking(heading, pitch, plan_loc, enemy_loc) / 180
        ]
        # print(df.get_plane_action(self.planeID))
        df.update_scene()

        return ob, reward, terminate, {}


    def reward(self):
        # if self.terminate() == 1:
        #     reward = 50
        if df.get_plane_state(self.enemyID)["health_level"] <= 0.9:
            reward = 50
        elif self.terminate() == -1:
            reward = -50
        else:
            reward = 0
        if df.get_plane_state(self.enemyID)['target_locked']:
            reward -= 10
        # reward += (3-self.missle_count)*0.01
        # reward += 50 / self.getDistance()
        # reward -= self.getDistance() / 10000
        if df.get_plane_state(self.planeID)['target_locked']:
            reward += 0.05
        return reward
    def terminate(self):
        '''
        0 ： 正常情况
        1 ： 敌机血量小于 0.8 战胜
        2 ： 单回合达到1000步
        -1 ： 自己被击中或者坠机 失败
        '''
        if self.step_game >= 2048:
            return 2
        # 判定HP首先小于80的人战败
        if self.getHP() <= .8:
            return -1
        elif df.get_plane_state(self.enemyID)["health_level"] <= 0.9:
            return 1
        else:
            return 0

    def getDistance(self):
        return ((df.get_plane_state(self.planeID)['position'][0] - df.get_plane_state(self.enemyID)['position'][0]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][1] - df.get_plane_state(self.enemyID)['position'][1]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][2] - df.get_plane_state(self.enemyID)['position'][2]) ** 2) ** .5

    def reset(self):
        self.__init__(self.host, self.port, self.rendering)

        heading = df.get_plane_state(self.planeID)['heading']
        pitch = df.get_plane_state(self.planeID)['pitch_attitude']
        plan_loc = df.get_plane_state(self.planeID)['position']
        enemy_loc = df.get_plane_state(self.enemyID)['position']
        ob = [  # normalized
            plan_loc[0] / 100,
            plan_loc[2] / 100,
            (plan_loc[1] - 1000) / 100,
            heading / 90,
            pitch / 90,
            df.get_plane_state(self.planeID)['roll_attitude'] / 90,
            enemy_loc[0] / 100,
            enemy_loc[2] / 100,
            (enemy_loc[1] - 1000) / 100,
            df.get_plane_state(self.enemyID)['target_locked'],
            df.get_plane_state(self.enemyID)['target_locked'],
            self.getDistance()/1000,
            self.getEnemyHP(),
            self.angle_attacking(heading, pitch, plan_loc, enemy_loc) / 180
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
        if action[3] > .8 and self.missle_count < 3:
            self.missle_count += 1
            df.fire_missile(self.planeID, self.missle_count)

        df.set_plane_pitch(self.planeID2, float(action[0]))
        df.set_plane_roll(self.planeID2, float(action[1]))
        df.set_plane_thrust(self.planeID2, float(action[2]))
        if action[3] > .8 and self.missle_count < 3:
            df.fire_missile(self.planeID2, self.missle_count)

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