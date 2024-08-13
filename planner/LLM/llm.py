from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import pandas as pd

from planner.plannerBase import PlannerBase
from utils.observation import Observation
from rich import print

class LLMController(PlannerBase):
    def __init__(self,
                sce,
                DA,
                a_bound = 3.0,
                DELTA_T = 0.04,
                # DELTA_SPEED = 0.5,
                length = 4.924,
                width = 1.872,
                max_steering_angle = 0.5)-> None:
        self.sce = sce
        self.DA = DA
        self.a_bound = a_bound
        self.DELTA_SPEED = DELTA_T * a_bound
        self.DELTA_T = DELTA_T
        self.length = length
        self.width = width
        self.max_steering_angle = max_steering_angle
        self.previous_heading_error = 0.0
        self.bias = 1.5


    def init(self, scenario_info: dict) -> None:
        print("----------------------------LLM_Dilu INIT----------------------------")
        print(scenario_info)
        print("-------------------------ZYH is so handsome-------------------------")

    def act(self, observation:Observation) -> List:
        # 主车信息
        frame = pd.DataFrame(
            vars(observation.ego_info),
            columns=['x', 'y', 'v', 'yaw', 'length', 'width'],
            index=['ego']
        )
        self.speed = frame['v']['ego']
        self.target_speed = frame['v']['ego']
        self.yaw = frame['yaw']['ego']
        self.y = frame['y']['ego']
        self.target_lane_index = self.locate_lane_index(self.y)
        # self.v_x = frame['v']['ego'] * np.cos(frame['yaw']['ego'])
        # self.v_y = frame['v']['ego'] * np.sin(frame['yaw']['ego'])
        ACTIONS_ALL = {
            0: 'LANE_LEFT',
            1: 'IDLE',
            2: 'LANE_RIGHT',
            3: 'FASTER',
            4: 'SLOWER'
        }
        delta_yaw = 1 * np.pi / 180 # 允许车辆在直行时有6°偏差
        if abs(self.yaw - np.pi) < np.pi - delta_yaw:
            # 说明车辆正在转向过程中
            self.target_lane_index = self.locate_target_lane_index(self.y, self.yaw)
        else:
            # 车辆基本回正

            # 获取llm返回的决定
            result = self.get_llm_result(observation)
            # print("转弯啦转弯啦！！")
            if ACTIONS_ALL[result] == "FASTER":
                self.target_speed += self.DELTA_SPEED
                self.target_lane_index = self.locate_target_lane_index(self.y, self.yaw)
            elif ACTIONS_ALL[result] == "SLOWER":
                self.target_speed -= self.DELTA_SPEED
                self.target_lane_index = self.locate_target_lane_index(self.y, self.yaw)
            elif ACTIONS_ALL[result] == "LANE_LEFT":
                self.target_lane_index = min(self.locate_lane_index(frame['y']['ego']) + 1, 2)
            elif ACTIONS_ALL[result] == "LANE_RIGHT":
                self.target_lane_index = max(self.locate_lane_index(frame['y']['ego']) - 1, 0)

        acc_target = self.speed_control(self.target_speed)
        wheel_target = self.steering_control(v = self.speed,
                                             y = frame['y']['ego'],
                                             yaw = frame['yaw']['ego'],
                                             target_y = self.lane_id_to_y(self.target_lane_index) # -3,-7,-11
                                             # target_y = -7
                                             )
        drive_control = [acc_target, wheel_target]  # 最终返回控制信息
        return drive_control

    def get_llm_result(self, observation) -> int:

        action = 'Not available'

        fewshot_messages = []
        fewshot_answers = []
        print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")

        frame = pd.DataFrame(
            vars(observation.ego_info),
            columns=['a', 'x', 'y', 'v', 'yaw', 'length', 'width'],
            index=['ego']
        )
        # 加载背景要素状态信息
        sub_frame = pd.DataFrame(columns=['x', 'y', 'v', 'a', 'yaw', 'length', 'width'])
        for obj_type in observation.object_info:
            for obj_name, obj_info in observation.object_info[obj_type].items():
                sub_sub_frame = pd.DataFrame(vars(obj_info), columns=['x', 'y', 'v', 'a', 'yaw', 'length', 'width'],
                                             index=[obj_name])
                sub_frame = pd.concat([sub_frame, sub_sub_frame])

        sce_descrip = self.sce.describe(frame, sub_frame)
        avail_action = self.sce.availableActionsDescription(frame)
        # print('[cyan]Scenario description: [/cyan]\n', sce_descrip)
        # print('[cyan]Available actions: [/cyan]\n',avail_action)

        action, response, human_question, fewshot_answer = self.DA.few_shot_decision(
            scenario_description=sce_descrip, available_actions=avail_action,
            previous_decisions=action,
            fewshot_messages=fewshot_messages,
            driving_intensions="Drive safely and avoid collisions, but do not exceed 33m/s. Stay in the LEFTMOST lane if possible! ",
            fewshot_answers=fewshot_answers,
        )

        return action

    def locate_target_lane_index(self, y, yaw) -> int:
        present_lane_index = self.locate_lane_index(y)
        if yaw >= 1 * np.pi / 180 and yaw <= np.pi/2:
            if y > self.lane_id_to_y(present_lane_index):
                target_lane_id = min(present_lane_index + 1, 2)
            elif y <= self.lane_id_to_y(present_lane_index):
                target_lane_id = present_lane_index
        elif yaw >= 3 * np.pi/2 and yaw <= (2 * np.pi-1 * np.pi / 180):
            if y > self.lane_id_to_y(present_lane_index):
                target_lane_id = present_lane_index
            elif y <= self.lane_id_to_y(present_lane_index):
                target_lane_id = max(present_lane_index - 1, 0)
        else:
            target_lane_id = present_lane_index
        return target_lane_id

    def lane_id_to_y(self, target_lane_id: int) -> int:
        # 从下到上的lane_id分别是0, 1, 2
        if target_lane_id == 0:
            target_y = -11-self.bias
        elif target_lane_id == 1:
            target_y = -7-self.bias
        elif target_lane_id == 2:
            target_y = -3-self.bias
        else:
            raise "目标车道不存在，请重新输入"
        return target_y

    def locate_lane_index(self, real_y):
        # 定义目标列表
        lane_coords = [-3, -7, -11]
        lane_coords = [x - self.bias for x in lane_coords]

        # 使用min函数和abs函数找到最接近的值
        closest_lane_y = min(lane_coords, key=lambda x: abs(x - real_y))
        if closest_lane_y == -3 - self.bias:
            closest_lane_index = 2
        elif closest_lane_y == -7 - self.bias:
            closest_lane_index = 1
        elif closest_lane_y == -11 - self.bias:
            closest_lane_index = 0
        else:
            raise "不对不对"

        return closest_lane_index

    def speed_control(self, target_speed: float) -> float:
        return (target_speed - self.speed)/self.DELTA_T  # 加速度

    def steering_control(self, v, y, yaw, target_y):
        """
        控制车辆转向以跟随指定车道的中心线。

        :param v: 车辆速度 [m/s]
        :param y: 车辆当前位置的y坐标 [m]
        :param yaw: 车辆当前的偏角 [rad]
        :param target_y: 目标车道中心线的y值 [m]
        :return: 转向角度 [rad]
        """
        # 车辆参数和控制器参数
        TAU_LATERAL = 0.6  # 横向控制时间常数 [s]
        TAU_HEADING = 0.2  # 航向控制时间常数 [s]
        KP_LATERAL = 0.5 / TAU_LATERAL
        KP_HEADING = 0.5 / TAU_HEADING
        # KD_HEADING = 0.1  # 设定微分控制增益


        # 横向位置控制
        lateral_error = target_y - y  # 计算横向误差
        lateral_speed_command = KP_LATERAL * lateral_error

        # 横向速度转换为航向角度
        heading_command = np.arcsin(np.clip(lateral_speed_command / max(v, 1e-6), -1, 1))

        # 计算参考航向
        heading_ref = heading_command

        # 航向控制
        heading_error = heading_ref - yaw
        heading_rate_command = KP_HEADING * (np.arctan2(np.sin(heading_error), np.cos(heading_error)))
        # 航向控制（包括微分控制）
        # heading_error = heading_ref - yaw
        # heading_rate_derivative = KD_HEADING * (heading_error - self.previous_heading_error) / self.DELTA_T
        # heading_rate_command = KP_HEADING * (
        #     np.arctan2(np.sin(heading_error), np.cos(heading_error))) + heading_rate_derivative

        # 计算转向角
        slip_angle = np.arcsin(
            np.clip(self.length / 2 / max(v, 1e-6) * heading_rate_command, -1, 1)
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        # 限制转向角
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        # self.previous_heading_error = heading_error  # 保存当前误差供下次使用
        return float(steering_angle)







