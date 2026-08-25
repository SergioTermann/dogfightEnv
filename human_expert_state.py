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
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
import harfang as hg
from random import uniform
from math import radians
from datetime import datetime
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


class human_expert_state_env(Env):
    def __init__(self, host='10.134.100.116', port='50888', rendering=True) -> None:
        self.host = host
        self.port = port
        self.rendering = rendering
        self.step_game = 0  #给本局设定结束条件，初定500步
        self.missle_count = 0 #记录导弹的数量，发射的越多罚分越多

        self.old_obs = None
        self.total_step = 0
        self.buffer_size = 200000
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
        df.set_plane_thrust(self.enemyID, 1)
        df.set_plane_linear_speed(self.planeID, 85)
        df.set_plane_linear_speed(self.enemyID, 85)
        df.set_client_update_mode(True)
        df.get_targets_list(self.planeID)
        if self.rendering:
            df.set_renderless_mode(False)
        else:
            df.set_renderless_mode(True)
        #收起起落架
        df.retract_gear(self.planes[0])
        # self.action_space_controller = Box(
        #     low=np.array([
        #         -1,  # Roll 俯仰角
        #         -1,  # Pitch 翻滚角
        #         -1,  # heading 航向
        #          0,  # linear speed 空速
        #          0,  # fire count 发射导弹数量
        #     ]),
        #     high=np.array([
        #         1,
        #         1,
        #         1,
        #         1,
        #         1,
        #     ]),
        # )
        self.action_space_state = Box(
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
        self.replay_buffer_state = ReplayBuffer(self.buffer_size, self.observation_space, self.action_space_state)
        # self.replay_buffer_controller = ReplayBuffer(self.buffer_size, self.action_space_controller, self.action_space_controller) #把观测也做成action
    def getDistance(self):
        return ((df.get_plane_state(self.planeID)['position'][0] - df.get_plane_state(self.enemyID)['position'][0]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][1] - df.get_plane_state(self.enemyID)['position'][1]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][2] - df.get_plane_state(self.enemyID)['position'][2]) ** 2) ** .5

    def render(self, id=0):
        df.set_renderless_mode(False)

    def step(self):
        self.step_game += 1
        self.total_step += 1

        if self.step_game % 200 == 0:
            df.set_plane_pitch(self.enemyID, np.random.random()-0.5)
            df.set_plane_roll(self.enemyID, np.random.random()-0.5)
        action_get = df.get_gamepad_action(self.planeID, False)
        # action_get = df.get_plane_action(self.planeID)
        old_action = [action_get["ROLL"], action_get["PITCH"], action_get["YAW"], action_get["THRUST"], action_get["FIRE"]/3]

        # for i in range(100):
        df.update_scene()
        while True:
            flag = df.get_finish_flag()
            if flag:
                break

        new_reward = self.reward()
        terminate = True if self.terminate() else False
        new_plane_state = df.get_plane_state(self.planeID)
        new_plan_loc = new_plane_state['position']
        new_enemy_loc = df.get_plane_state(self.enemyID)['position']
        action_get = df.get_gamepad_action(self.planeID, False)

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
            self.facing/180,
            self.aming/90,
        ]
        if self.old_obs is None:
            self.old_obs = new_ob
        old_pos = [
            self.old_obs[3],
            self.old_obs[4],
            self.old_obs[5],
            self.old_obs[8],
            self.old_obs[7],
        ]
        target_pos = [
            new_plane_state['roll_attitude'] / 90,
            new_plane_state['pitch_attitude'] / 90,
            new_plane_state['heading'] / 360,
            new_plane_state['linear_speed'] / 1000,
            action_get["FIRE"]/3,
        ]
        old_action = np.array(old_action)
        target_pos = np.array(target_pos)
        old_pos = np.array(old_pos)
        # self.replay_buffer_controller.add(old_pos, target_pos, old_action, new_reward, terminate, [{}])
        self.replay_buffer_state.add(self.old_obs, new_ob, old_action, new_reward, terminate, [{}])
        self.old_obs = new_ob
        if self.replay_buffer_state.full:
            now = datetime.now()
            now = now.strftime("%Y_%m_%d_%H_%M_%S")
            path_states = "D:\desktop\code\save_buffer\states"+"\\" + now + ".pkl"
            # path_controller = "D:\desktop\code\save_buffer\controller" + "\\" + now + ".pkl"
            save_to_pkl(path_states, self.replay_buffer_state)
            # save_to_pkl(path_controller, self.replay_buffer_controller)
            self.replay_buffer_state = ReplayBuffer(self.buffer_size, self.observation_space, self.action_space_state)
            # self.replay_buffer_controller = ReplayBuffer(self.buffer_size, self.action_space_controller,
            #                                              self.action_space_controller)  # 把观测也做成action
            terminate = 1
        return new_ob, new_reward, terminate, {}

    def terminate(self):
        if self.step_game >= 4096:
            return 2
        elif df.get_plane_state(self.enemyID)["health_level"] <= 0.9:
            return 1
        else:
            return

    def reset(self):
        print(self.total_step)
        self.step_game = 0  # 给本局设定结束条件，初定500步
        self.missle_count = 0 # 记录导弹的数量，发射的越多罚分越多
        for i in self.planes:
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
        df.retract_gear(self.planes[0])
        new_plane_state = df.get_plane_state(self.planeID)
        new_plan_loc = plane_state['position']
        new_enemy_loc = df.get_plane_state(self.enemyID)['position']
        df.get_targets_list(self.planeID)
        df.get_gamepad_action(self.planeID, True)
        new_enemy_state = df.get_plane_state(self.enemyID)

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
            self.facing/180,
            self.aming/90,
        ]

        self.last_obs = ob
        self.missle_count = 0
        self.step_game = 0

        range = hg.Vec3(500, 500, 500)
        center = hg.Vec3(1000, 2000, 3000)
        y_orientations_range = hg.Vec2(-45, 45)
        df.reset_machine_matrix(self.enemyID,
                                uniform(center.x - range.x / 2, center.x + range.x / 2),
                                uniform(center.y - range.y / 2, center.y + range.y / 2),
                                uniform(center.z - range.z / 2, center.z + range.z / 2),
                                0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)

        center = hg.Vec3(1000, 2000, 1500)
        range_plane = hg.Vec3(0, 0, 0)
        y_orientations_range = hg.Vec2(-45, 45)
        df.reset_machine_matrix(self.planeID,
                                uniform(center.x - range_plane.x / 2, center.x + range_plane.x / 2),
                                uniform(center.y - range_plane.y / 2, center.y + range_plane.y / 2),
                                uniform(center.z - range_plane.z / 2, center.z + range_plane.z / 2),
                                0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0)

        df.set_plane_linear_speed(self.planeID, 85)
        df.set_plane_linear_speed(self.enemyID, 85)

        return ob

    def reward(self):
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