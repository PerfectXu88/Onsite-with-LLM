## 	LLM规控器说明

### 1 快速开始

#### 1.1 系统环境配置

> 注：目前仅支持windows操作系统及Ubuntu 20.04两种系统环境

1. 使用conda创建虚拟环境，指定版本为python3.8

   ```python
   conda create -n onsite_dilu python=3.8
   ```

2. 激活虚拟环境

   ```python
   conda activate onsite_dilu
   ```

3. 载入依赖库

   ```python
   pip install -r requirements.txt
   ```

   ✍️Note：Dilu需要某些特定版本的python依赖库，例如``langchain==0.0.335``，``openai==0.28.1``，``chromadb==0.3.29``，因此请采用**requirements.txt**中的要求。

#### 1.2 环境变量配置

> 环境变量配置位于``./config/config.yaml``和``./config/task.yaml``中

> 由于本项目使用了OpenAI提供的API key，因此需要使用🪄魔法，在项目中使用魔法需要环境变量进行一定配置。

- 在``main.py``中的``set_up()``函数，使用``os``库对OPENAI所需要的环境变量进行设置，主要包括``OPENAI_API_TYPE``,``OPENAI_API_KEY``,``OPENAI_CHAT_MODEL``等等。

- ``set_up()``函数：

```python
def setup_env(config):
    if config['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
        os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
        proxy_url = config['PROXY_URL']
        proxy_port = config['PROXY_PORT']
        os.environ["http_proxy"] = f'{proxy_url}:{proxy_port}'
        os.environ["https_proxy"] = f'{proxy_url}:{proxy_port}'
    else:
        raise ValueError("Unknown OPENAI_API_TYPE, should be openai")
```

- 上述变量均在``./config/task.yaml``进行配置：

  1. ``OPENAI_API_TYPE``：``openai``

     > 本项目目前仅支持*‘openai’*或者*'azure'*，本文档以openai为例

  2. ``OPENAI_KEY``：‘sk-xxxxxx’

     > 专用密钥需要自行申请，申请方式见[3.1 密钥申请](# 3.1 密钥申请)

  3. ``OPENAI_CHAT_MODEL``：``'gpt-3.5-turbo-16k'``

     > ``gpt-4-1106-preview``、``'gpt-3.5-turbo-16k'``等，详见[OpenAI官网](https://platform.openai.com/)

  4. ``PROXY_URL``：127.0.0.1

     > 本地端口

  5. ``PROXY_PORT``：1234

     > 在电脑设置-网络和Internet-代理-手动代理设置-编辑中找到代理IP地址和端口，分别对应``PROXY_URL``和``PROXY_PORT``

#### 1.3 规控算法指定

- 回放测试模块

  > 由于Dilu与onsite所需TESSNG所需要的python版本不兼容，因此本项目暂时仅能使用回放测试功能，该问题将会在后续版本修复

  注释掉``main.py``文件中以下语句，即可仅使用回放测试功能：

  ```python
  import TessNG
  ```

  ```python
  TessNG.run(mode, {})
  ```

  ```python
  TessNG.run(mode, config, PLANNER(), scene_info=scenario_manager.cur_scene)
  ```

  其中，若注释掉之后项目无法正常运行，请将该语句替换为``pass``

  在``./config/task.yaml``中，可以采用如下配置：

  ```yaml
  REPLAY:
    tasks:
    visualize: True
    skipExist: False
  ```

- 选择待测试的规控算法

  规控算法放置路径：``planner/``

  通过``planner/__init__``中的``planner``字段实现对规控算法的指定

  - Dilu算法：

    ```python
    from planner.LLM.llm import LLMController
    PLANNER = LLMController
    ```


### 2 LLM决策控制

#### 2.1 观察空间对齐

- 首先需要对环境中背景交通流信息提取

  ```python
  frame = pd.DataFrame(
              vars(observation.ego_info),
              columns=['a', 'x', 'y', 'v', 'yaw', 'length', 'width'],
              index=['ego']
          )
          sub_frame = pd.DataFrame(columns=['x', 'y', 'v', 'a', 'yaw', 'length', 'width'])
          # 加载背景要素状态信息
          for obj_type in observation.object_info:
              for obj_name, obj_info in observation.object_info[obj_type].items():
                  sub_obj_frame = pd.DataFrame(vars(obj_info), columns=['x', 'y', 'v', 'a', 'yaw', 'length', 'width'],
                                           index=[obj_name])
                  sub_frame = pd.concat([sub_frame, sub_obj_frame])
  ```

- 其中，``frame``中存储信息为主车信息，``sub_frame``中存储的信息为环境车流及障碍物信息

- 上述信息分别通过``EnvScenario``类内的``describe()``和``availableActionsDescription()``函数生成对场景的描述并获取可行动作

  ``describe()``函数：

  ```python
  def describe(self, frame, subframe) -> str:
  
      currentLaneIndex = self.get_lane_index(frame['y']['ego'])
      roadCondition = self.processNormalLane(currentLaneIndex, frame)
      SVDescription = self.describeSVNormalLane(frame['x']['ego'], currentLaneIndex, frame, subframe)
  
      return roadCondition + SVDescription
  ```

  ``processNormalLane()``函数：

  ```python
  def processNormalLane(self, LaneIndex, frame) -> str:
  
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
  
      return description
  ```

  ``describeSVNormalLane``函数：

  ```python
  def describeSVNormalLane(self, ego_pos, LaneIndex, frame, subframe) -> str:
      surroundVehicles = subframe
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
  ```

- 通过上述代码生成环境和主车信息描述，利用ChatGPT获取回复

- ``DriverAgent``可以利用LLM获取回复，主要使用``DriverAgent``类内的``few_shot_decision``函数实现

  ``few_shot_decision()``函数：

  ```python
  def few_shot_decision(self, scenario_description: str = "Not available", previous_decisions: str = "Not available", available_actions: str = "Not available", driving_intensions: str = "Not available", fewshot_messages: List[str] = None, fewshot_answers: List[str] = None):
  
      system_message = textwrap.dedent(f"""\
      You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
      You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.
  
      Your response should use the following format:
      <reasoning>
      <reasoning>
      <repeat until you have a decision>
      Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 
  
      Make sure to include {delimiter} to separate every step.
      """)
  
      human_message = f"""\
      Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 
  
      Here is the current scenario:
      {delimiter} Driving scenario description:
      {scenario_description}
      {delimiter} Driving Intensions:
      {driving_intensions}
      {delimiter} Available actions:
      {available_actions}
  
      You can stop reasoning once you have a valid action to take. 
      """
      human_message = human_message.replace("        ", "")
      messages = [
          SystemMessage(content=system_message),
      ]
      messages.append(
          HumanMessage(content=human_message)
      )
      # print("fewshot number:", (len(messages) - 2)/2)
      start_time = time.time()
      # with get_openai_callback() as cb:
      # response = self.llm(messages)
      # print(response.content)
      print("[cyan]Agent answer:[/cyan]")
      response_content = ""
      for chunk in self.llm.stream(messages):
          response_content += chunk.content
          print(chunk.content, end="", flush=True)
      print("\n")
      decision_action = response_content.split(delimiter)[-1]
      try:
          result = int(decision_action)
          if result < 0 or result > 4:
              raise ValueError
      except ValueError:
          print("Output is not a int number, checking the output...")
          check_message = f"""
          You are a output checking assistant who is responsible for checking the output of another agent.
  
          The output you received is: {decision_action}
  
          Your should just output the right int type of action_id, with no other characters or delimiters.
          i.e. :
          | Action_id | Action Description                                     |
          |--------|--------------------------------------------------------|
          | 0      | Turn-left: change lane to the left of the current lane |
          | 1      | IDLE: remain in the current lane with current speed   |
          | 2      | Turn-right: change lane to the right of the current lane|
          | 3      | Acceleration: accelerate the vehicle                 |
          | 4      | Deceleration: decelerate the vehicle                 |
  
  
          You answer format would be:
          {delimiter} <correct action_id within 0-4>
          """
          messages = [
              HumanMessage(content=check_message),
          ]
          with get_openai_callback() as cb:
              check_response = self.llm(messages)
          result = int(check_response.content.split(delimiter)[-1])
  
      few_shot_answers_store = ""
      for i in range(len(fewshot_messages)):
          few_shot_answers_store += fewshot_answers[i] + \
              "\n---------------\n"
      print("Result:", result)
      return result, response_content, human_message, few_shot_answers_store
  ```

#### 2.2 动作空间对齐

- 通过LLM可以获取高级决策结果，由于LLM对数据敏感性较低，因此主要采用离散动作空间表述

  离散动作空间：

  ```python
  ACTIONS_ALL = {
              0: 'LANE_LEFT',
              1: 'IDLE',
              2: 'LANE_RIGHT',
              3: 'FASTER',
              4: 'SLOWER'
          }
  ```

- 其中，LLM仅在车辆回正之后进行决策，在转弯过程中不进行决策。同时为了保证车辆在连续多个时间步中实现完整的转弯动作，需要进行车辆动作姿态的判定。

- 动作决策应分别返回车辆的加速度和转向角控制，并由底层车辆控制器完成在仿真环境中的移动。

  ``act()``函数：

  ```python
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
  ```

- 其中，车辆变道动作由函数独立控制

  ``steering_control()``函数：

  ```python
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
      KP_LATERAL = 1 / TAU_LATERAL
      KP_HEADING = 1 / TAU_HEADING
  
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
  
      # 计算转向角
      slip_angle = np.arcsin(
          np.clip(self.length / 2 / max(v, 1e-6) * heading_rate_command, -1, 1)
      )
      steering_angle = np.arctan(2 * np.tan(slip_angle))
  
      # 限制转向角
      steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
  
      return float(steering_angle)
  ```

#### 2.3 Dilu架构功能

- 目前本项目仅添加了Dilu的思考模块，反思模块将会在后续进一步更新
- Dilu架构及功能详见[Dilu](https://github.com/PJLab-ADG/DiLu)

### 3 网络问题

#### 3.1 密钥申请

- 首先，想要申请Openai的key，必须先有一个Openai的账号

- 其次，需要一份由海外发行的银行卡（Visa或者万事达），国内或者港澳发行的银行卡均不可用，但是可以用虚拟卡替代

- 登录之后，在[API管理平台](https://platform.openai.com/api-keys)点击如图所示按钮即可申请你自己的API-key了

  ![image-20240806171548753](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20240806171548753.png)

- 获取key之后，在项目``config/config.yaml``文件中修改即可


