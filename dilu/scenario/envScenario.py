from typing import List, Tuple, Optional, Union, Dict
from datetime import datetime
import math
import os

from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.road.lane import (
    StraightLane, CircularLane, SineLane, PolyLane, PolyLaneFixedWidth
)
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle

import pandas as pd
import numpy as np
from utils.observation import Observation


ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'Turn-left - change lane to the left of the current lane',
    1: 'IDLE - remain in the current lane with current speed',
    2: 'Turn-right - change lane to the right of the current lane',
    3: 'Acceleration - accelerate the vehicle',
    4: 'Deceleration - decelerate the vehicle'
}


class EnvScenario:
    def __init__(self) -> None:
        pass

    def availableActionsDescription(self, frame) -> str:
        LaneIndex = self.get_lane_index(frame['y']['ego'])

        if LaneIndex == 0:
            ava = [1, 2, 3, 4]
        elif LaneIndex == 2:
            ava = [0, 1, 3, 4]
        else:
            ava = [0, 1, 2, 3, 4]

        avaliableActionDescription = 'Your available actions are: \n'

        for action in ava:
            avaliableActionDescription += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(
                action) + '\n'
        # if 1 in availableActions:
        #     avaliableActionDescription += 'You should check IDLE action as FIRST priority. '
        # if 0 in availableActions or 2 in availableActions:
        #     avaliableActionDescription += 'For change lane action, CAREFULLY CHECK the safety of vehicles on target lane. '
        # if 3 in availableActions:
        #     avaliableActionDescription += 'Consider acceleration action carefully. '
        # if 4 in availableActions:
        #     avaliableActionDescription += 'The deceleration action is LAST priority. '
        # avaliableActionDescription += '\n'
        return avaliableActionDescription

    def processNormalLane(self, LaneIndex, frame) -> str:

        # print(frame)
        acce = 0 if pd.isna(frame['a']['ego']) else frame['a']['ego']
        x = frame['x']['ego']
        y = frame['y']['ego']
        speed = frame['v']['ego']

        #获取车辆位置信息的描述
        numLanes = 3

        egoLaneRank = LaneIndex

        if egoLaneRank == 0:
            description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the leftmost lane. "
        elif egoLaneRank == 2:
            description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the rightmost lane. "
        else:
            description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the second lane from the left. "

        description += f"Your current position is `({x:.2f}, {y:.2f})`, speed is {speed:.2f} m/s, acceleration is {acce:.2f} m/s^2"
        # 这个lane position感觉暂时没用 ", and lane position is {self.getLanePosition(self.ego):.2f} m.\n"
        return description

    def getVehDis(self, veh: IDMVehicle):
        posA = self.ego.position
        posB = veh.position
        distance = np.linalg.norm(posA - posB)
        return distance

    def getClosestSV(self, SVs: List[IDMVehicle]):
        if SVs:
            closestIdex = -1
            closestDis = 99999999
            for i, sv in enumerate(SVs):
                dis = self.getVehDis(sv)
                if dis < closestDis:
                    closestDis = dis
                    closestIdex = i
            return SVs[closestIdex]
        else:
            return None

    def processSingleLaneSVs(self, SingleLaneSVs: List[IDMVehicle]):
        # 返回当前车道上，前方最近的车辆和后方最近的车辆，如果没有，则为 None
        if SingleLaneSVs:
            aheadSVs = []
            behindSVs = []
            for sv in SingleLaneSVs:
                RSStr = self.getSVRelativeState(sv)
                if RSStr == 'is ahead of you':
                    aheadSVs.append(sv)
                else:
                    behindSVs.append(sv)
            aheadClosestOne = self.getClosestSV(aheadSVs)
            behindClosestOne = self.getClosestSV(behindSVs)
            return aheadClosestOne, behindClosestOne
        else:
            return None, None

    def processSVsNormalLane(
            self, SVs: List[IDMVehicle], currentLaneIndex: LaneIndex
    ):
        # 目前 description 中的车辆有些太多了，需要处理一下，只保留最靠近 ego 的几辆车
        classifiedSVs: Dict[str, List[IDMVehicle]] = {
            'current lane': [],
            'left lane': [],
            'right lane': [],
            'target lane': []
        }
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        for sv in SVs:
            lidx = sv.lane_index
            if lidx in sideLanes:
                if lidx == currentLaneIndex:
                    classifiedSVs['current lane'].append(sv)
                else:
                    laneRelative = lidx[2] - currentLaneIndex[2]
                    if laneRelative == 1:
                        classifiedSVs['right lane'].append(sv)
                    elif laneRelative == -1:
                        classifiedSVs['left lane'].append(sv)
                    else:
                        continue
            elif lidx == nextLane:
                classifiedSVs['target lane'].append(sv)
            else:
                continue

        validVehicles: List[IDMVehicle] = []
        existVehicles: Dict[str, bool] = {}
        for k, v in classifiedSVs.items():
            if v:
                existVehicles[k] = True
            else:
                existVehicles[k] = False
            ahead, behind = self.processSingleLaneSVs(v)
            if ahead:
                validVehicles.append(ahead)
            if behind:
                validVehicles.append(behind)

        return validVehicles, existVehicles

    def get_lane_index(self, real_y):
        # 定义目标列表
        lane_coords = [-3, -7, -11]

        # 使用min函数和abs函数找到最接近的值
        closest_lane_y = min(lane_coords, key=lambda x: abs(x - real_y))
        if closest_lane_y == -3:
            closest_lane_index = 0
        elif closest_lane_y == -7:
            closest_lane_index = 1
        elif closest_lane_y == -11:
            closest_lane_index = 2
        else:
            raise "不对不对"

        return closest_lane_index

    def getSurrendVehicles(self, subframe) -> list:
        SV = {}
        i = 0
        for car in subframe.index():
            SV[car] = {
                'lane_index': self.get_lane_index(car),
                'mark': i,
                'x': subframe['x'][car],
                'y': subframe['y'][car],
                'a': subframe['a'][car],
                'v': subframe['v'][car]
            }
        return

    def getSVRelativeState(self, ego_pos, sv_x) -> str:
        # CAUTION: 这里有一个问题，pygame 的 y 轴是上下颠倒的，向下是 y 轴的正方向。
        #       因此，在 highway-v0 上，车辆向左换道实际上是向右运动。因此判断车辆相
        #       对自车的位置，不能用向量来算，直接根据车辆在哪条车道上来判断是比较合适
        #       的，向量只能用来判断车辆在 ego 的前方还是后方
        relativePosition = sv_x - ego_pos

        # 看不懂是干啥的呃
        # egoUnitVector = self.getUnitVector(self.ego.heading)
        # cosineValue = sum(
        #     [x*y for x, y in zip(relativePosition, egoUnitVector)]
        # )

        if relativePosition >= 0: # 原来是 cosineValue >= 0
            return 'is ahead of you'
        else:
            return 'is behind of you'

    def describeSVNormalLane(self, ego_pos, LaneIndex, frame, subframe) -> str:
        # 当 ego 在 StraightLane 上时，车道信息是重要的，需要处理车道信息
        # 首先判断车辆是不是和车辆在同一条 road 上
        #   如果在同一条 road 上，则判断在哪条 lane 上
        #   如果不在同一条 road 上，则判断是否在 next_lane 上
        #      如果不在 nextLane 上，则直接不考虑这辆车的信息
        #      如果在 nextLane 上，则统计这辆车关于 ego 的相对运动状态

        # sideLanes = self.get_side_lanes(LaneIndex)
        # nextLane = self.network.next_lane(
        #     LaneIndex, self.ego.route, self.ego.position
        # )
        surroundVehicles = subframe
        # 非法检测被删除，有bug再加回来
        if surroundVehicles.empty:
            SVDescription = "There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n"
            return SVDescription
        else:
            SVDescription = ''
            for sv in surroundVehicles.index:
                sv_x = subframe['x'][sv]
                sv_y = subframe['y'][sv]
                sv_v = subframe['v'][sv]
                sv_a = subframe['a'][sv]
                lidx = self.get_lane_index(sv_y)
                mark = str(sv)[3]
                if lidx == LaneIndex:
                    SVDescription += f"- Vehicle `{mark}` is driving on the same lane as you and {self.getSVRelativeState(ego_pos, sv_x)}. "
                else:
                    laneRelative = lidx - LaneIndex
                    if laneRelative == 1:
                        # laneRelative = 1 表示车辆在 ego 的右侧车道上行驶
                        SVDescription += f"- Vehicle `{mark}` is driving on the lane to your right and {self.getSVRelativeState(ego_pos, sv_x)}. "
                    elif laneRelative == -1:
                        # laneRelative = -1 表示车辆在 ego 的左侧车道上行驶
                        SVDescription += f"- Vehicle `{mark}` is driving on the lane to your left and {self.getSVRelativeState(ego_pos, sv_x)}. "

                SVDescription += f"The position of it is `({sv_x:.2f}, {sv_y:.2f})`, speed is {sv_v:.2f} m/s, acceleration is {sv_a:.2f} m/s^2.\n" # , and lane position is {self.getLanePosition(sv):.2f} m

            if SVDescription:
                descriptionPrefix = "There are other vehicles driving around you, and below is their basic information:\n"
                return descriptionPrefix + SVDescription
            else:
                SVDescription = 'There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n'
                return SVDescription

    def describe(self, frame, subframe) -> str:

        currentLaneIndex = self.get_lane_index(frame['y']['ego'])
        roadCondition = self.processNormalLane(currentLaneIndex, frame)
        SVDescription = self.describeSVNormalLane(frame['x']['ego'], currentLaneIndex, frame, subframe)

        return roadCondition + SVDescription

    def promptsCommit(
        self, decisionFrame: int, vectorID: str, done: bool,
        description: str, fewshots: str, thoughtsAndAction: str
    ):
        self.dbBridge.insertPrompts(
            decisionFrame, vectorID, done, description,
            fewshots, thoughtsAndAction
        )
