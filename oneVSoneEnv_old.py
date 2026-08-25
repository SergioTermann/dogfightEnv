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
                -300,  # x / 100
                -300,  # y / 100
                -1,    # z / 50
                -360,  # roll_attitude * 4
                -360,  # pitch_attitude * 4
                 0,    # heading
                 0,    # thrust level 油门
                 0,    # missile count 导弹发射计数/3
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
                1,     # missile count 导弹发射计数/3
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
        return df.get_health(self.enemyID)['health_level']

    def render(self, id=0):
        df.set_renderless_mode(False)

    def step(self, action):
        self.step_game += 1
        self.total_step += 1
        self.sendAction(action)
        if self.step_game % 200 == 0:
            df.set_plane_pitch(self.enemyID, np.random.random()-0.5)
            df.set_plane_roll(self.enemyID, np.random.random()-0.5)
        # for i in range(100):
        df.update_scene()
        while True:
            flag = df.get_finish_flag()
            if flag:
                break

        reward = self.reward()
        terminate = True if self.terminate() else False
        new_plane_state = df.get_plane_state(self.planeID)
        new_plan_loc = new_plane_state['position']
        new_enemy_loc = df.get_plane_state(self.enemyID)['position']
        new_ob = [  # normalized
            new_plan_loc[0] / 1000,
            new_plan_loc[2] / 1000,
            new_plan_loc[1] / 1000,
            new_plane_state['roll_attitude'] / 90,
            new_plane_state['pitch_attitude'] / 90,
            new_plane_state['heading'] / 360,
            new_plane_state['thrust_level'],
            self.missle_count / 3,
            new_plane_state['linear_speed'] / 1000,
            new_plane_state['vertical_speed'] / 300,
            new_plane_state['horizontal_speed'] / 500,
            new_enemy_loc[0] / 1000,
            new_enemy_loc[2] / 1000,
            new_enemy_loc[1] / 1000,
            df.get_plane_state(self.enemyID)['target_locked'],
            df.get_plane_state(self.planeID)['target_locked'],
            self.getDistance() / 1000,
            self.getEnemyHP(),
            self.angle_attacking(new_plane_state['heading'], new_plane_state['pitch_attitude'], new_plan_loc,
                                 new_enemy_loc) / 180
        ]
        # if self.step_game % 200 == 0:
        #     table.field_names = ['参数','值']
        #     table.add_row(['x/1000', plan_loc[0]/1000])
        #     table.add_row(['y/1000', plan_loc[2] / 1000])
        #     table.add_row(['z/1000',plan_loc[1] / 1000])
        #     table.add_row(['roll_attitude * 4',plane_state['roll_attitude'] / 90])
        #     table.add_row(['pitch_attitude * 4',plane_state['pitch_attitude'] / 90])
        #     table.add_row(['heading/360',plane_state['heading'] / 360])
        #     table.add_row(['thrust',plane_state['thrust_level']])
        #     table.add_row(['brake',plane_state['brake_level']])
        #     table.add_row(['flaps',plane_state['flaps_level']])
        #     table.add_row(['空速/1000',plane_state['linear_speed']/1000])
        #     table.add_row(['垂直速度/300',plane_state['vertical_speed']/300])
        #     table.add_row(['水平速度/500',plane_state['horizontal_speed']/500])
        #     table.add_row(['enymy x/1000',enemy_loc[0] / 1000])
        #     table.add_row(['enemy y/1000',enemy_loc[2] / 1000])
        #     table.add_row(['enemy z/1000',enemy_loc[1]/ 1000])
        #     table.add_row(['enemy target lock',df.get_plane_state(self.enemyID)['target_locked']])
        #     table.add_row(['plane target lock',df.get_plane_state(self.planeID)['target_locked']])
        #     table.add_row(['distance /1000',self.getDistance()/1000])
        #     table.add_row(['enemy HP',self.getEnemyHP()])
        #     table.add_row(['angle/180',self.angle_attacking(plane_state['heading'] , plane_state['pitch_attitude'], plan_loc, enemy_loc) / 180])
        #     print(table)
        #     table.clear()
        # print(df.get_plane_action(self.planeID))
        self.facing = self.facing_angle(new_plane_state['heading'], new_plan_loc[0], new_plan_loc[2], new_enemy_loc[0], new_enemy_loc[2])
        self.aming = self.angle_attacking(new_plane_state['heading'], new_plane_state['pitch_attitude'], new_plan_loc,
                     new_enemy_loc)
        return new_ob, reward, terminate, {}

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
            reward += 0.1

        return reward

    def terminate(self):

        if self.step_game >= 2048:
            return 2
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
        planes = df.get_planes_list()
        self.step_game = 0  #给本局设定结束条件，初定500步
        self.missle_count = 0 #记录导弹的数量，发射的越多罚分越多
        for i in planes:
            df.reset_machine(i)
        df.set_plane_thrust(self.planeID, 1)
        df.set_plane_thrust(self.enemyID, 1)
        df.set_client_update_mode(True)
        if self.step_game % 200 == 0:
            # df.set_plane_pitch(self.enemyID, 2*np.random.random()-1)
            df.set_plane_pitch(self.enemyID, np.random.random()-0.5)
            # df.set_plane_roll(self.enemyID, 2*np.random.random()-1)
            df.set_plane_roll(self.enemyID, np.random.random()-0.5)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)
        # 收起起落架
        df.retract_gear(planes[0])
        plane_state = df.get_plane_state(self.planeID)
        plan_loc = plane_state['position']
        enemy_loc = df.get_plane_state(self.enemyID)['position']
        df.get_targets_list(self.planeID)
        df.get_gamepad_action(self.planeID, True)

        ob = [  # normalized
            plan_loc[0] / 1000,
            plan_loc[2] / 1000,
            plan_loc[1] / 1000,
            plane_state['roll_attitude'] / 90,
            plane_state['pitch_attitude'] / 90,
            plane_state['heading'] / 360,
            plane_state['thrust_level'],
            self.missle_count,
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
        # if self.total_step < 500000:
        #     range = hg.Vec3(0, 0, 0)
        #     center = hg.Vec3(1000, 2000, 3000)
        #     y_orientations_range = hg.Vec2(-45, 45)
        #     df.reset_machine_matrix(self.enemyID,
        #                             uniform(center.x - range.x / 2, center.x + range.x / 2),
        #                             uniform(center.y - range.y / 2, center.y + range.y / 2),
        #                             uniform(center.z - range.z / 2, center.z + range.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        #
        #     center = hg.Vec3(1000, 2000, 1500)
        #     range_plane = hg.Vec3(0, 0, 0)
        #     y_orientations_range = hg.Vec2(-0, 0)
        #     df.reset_machine_matrix(self.planeID,
        #                             uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
        #                             uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
        #                             uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        # elif self.total_step < 1000000:
        #     range = hg.Vec3(100, 100, 100)
        #     center = hg.Vec3(1000, 2000, 3000)
        #     y_orientations_range = hg.Vec2(-45, 45)
        #     df.reset_machine_matrix(self.enemyID,
        #                             uniform(center.x - range.x / 2, center.x + range.x / 2),
        #                             uniform(center.y - range.y / 2, center.y + range.y / 2),
        #                             uniform(center.z - range.z / 2, center.z + range.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        #
        #     center = hg.Vec3(1000, 2000, 1500)
        #     range_plane = hg.Vec3(0, 0, 0)
        #     y_orientations_range = hg.Vec2(-0, 0)
        #     df.reset_machine_matrix(self.planeID,
        #                             uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
        #                             uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
        #                             uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        # elif self.total_step < 1500000:
        #     range = hg.Vec3(200, 200, 200)
        #     center = hg.Vec3(1000, 2000, 3000)
        #     y_orientations_range = hg.Vec2(-45, 45)
        #     df.reset_machine_matrix(self.enemyID,
        #                             uniform(center.x - range.x / 2, center.x + range.x / 2),
        #                             uniform(center.y - range.y / 2, center.y + range.y / 2),
        #                             uniform(center.z - range.z / 2, center.z + range.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        #
        #     center = hg.Vec3(1000, 2000, 1500)
        #     range_plane = hg.Vec3(100, 100, 100)
        #     y_orientations_range = hg.Vec2(-15, 15)
        #     df.reset_machine_matrix(self.planeID,
        #                             uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
        #                             uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
        #                             uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        # elif self.total_step < 2000000:
        #     range = hg.Vec3(200, 200, 200)
        #     center = hg.Vec3(1000, 2000, 3000)
        #     y_orientations_range = hg.Vec2(-45, 45)
        #     df.reset_machine_matrix(self.enemyID,
        #                             uniform(center.x - range.x / 2, center.x + range.x / 2),
        #                             uniform(center.y - range.y / 2, center.y + range.y / 2),
        #                             uniform(center.z - range.z / 2, center.z + range.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        #
        #     center = hg.Vec3(1000, 2000, 1500)
        #     range_plane = hg.Vec3(200, 200, 200)
        #     y_orientations_range = hg.Vec2(-30, 30)
        #     df.reset_machine_matrix(self.planeID,
        #                             uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
        #                             uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
        #                             uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        # elif self.total_step < 2500000:
        #     range = hg.Vec3(300, 300, 300)
        #     center = hg.Vec3(1000, 2000, 3000)
        #     y_orientations_range = hg.Vec2(-45, 45)
        #     df.reset_machine_matrix(self.enemyID,
        #                             uniform(center.x - range.x / 2, center.x + range.x / 2),
        #                             uniform(center.y - range.y / 2, center.y + range.y / 2),
        #                             uniform(center.z - range.z / 2, center.z + range.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        #
        #     center = hg.Vec3(1000, 2000, 1500)
        #     range_plane = hg.Vec3(300, 300, 300)
        #     y_orientations_range = hg.Vec2(-45, 45)
        #     df.reset_machine_matrix(self.planeID,
        #                             uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
        #                             uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
        #                             uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        # else:
        #     range = hg.Vec3(500, 500, 500)
        #     center = hg.Vec3(1000, 2000, 3500)
        #     y_orientations_range = hg.Vec2(-45, 45)
        #     df.reset_machine_matrix(self.enemyID,
        #                             uniform(center.x - range.x / 2, center.x + range.x / 2),
        #                             uniform(center.y - range.y / 2, center.y + range.y / 2),
        #                             uniform(center.z - range.z / 2, center.z + range.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)
        #
        #     center = hg.Vec3(1000, 2000, 1500)
        #     range_plane = hg.Vec3(500, 500, 500)
        #     y_orientations_range = hg.Vec2(-45, 45)
        #     df.reset_machine_matrix(self.planeID,
        #                             uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
        #                             uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
        #                             uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
        #                             0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)

        range = hg.Vec3(500, 500, 500)
        center = hg.Vec3(1000, 2000, 3000)
        y_orientations_range = hg.Vec2(-45, 45)
        df.reset_machine_matrix(self.enemyID,
                                uniform(center.x - range.x / 2, center.x + range.x / 2),
                                uniform(center.y - range.y / 2, center.y + range.y / 2),
                                uniform(center.z - range.z / 2, center.z + range.z / 2),
                                0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)

        range_plane = hg.Vec3(500, 500, 500)
        center = hg.Vec3(1000, 2000, 1500)
        y_orientations_range = hg.Vec2(-45, 45)
        df.reset_machine_matrix(self.planeID,
                                uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
                                uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
                                uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
                                0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)

        df.set_plane_linear_speed(self.planeID, 85)
        df.set_plane_linear_speed(self.enemyID, 85)
        return ob

    def sendAction(self, action, actionType=None):
        df.set_plane_roll(self.planeID, float(action[0]))
        # df.set_plane_roll(self.planeID, 10)
        df.set_plane_pitch(self.planeID, float(action[1]))
        df.set_plane_yaw(self.planeID, float(action[2]))
        # df.set_plane_flaps(self.planeID, float(action[3]))
        # df.set_plane_brake(self.planeID, float(action[4]))
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

    def facing_angle(self, heading, x, y, x_e, y_e):
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
        return result