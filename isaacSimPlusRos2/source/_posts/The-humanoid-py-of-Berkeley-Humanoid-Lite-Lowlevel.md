---
title: Berkeley Humanoid Lite Lowlevel 的 humanoid.py
date: 2026-02-10 20:25:21
tags: [Berkeley Humanoid Lite, Python]
categories: [Berkeley Humanoid Lite]
---

`humanoid.py` 运行在机器人机载计算机（如 Jetson Orin 或 NUC）上，负责连接真实的传感器和执行器，并管理机器人的运行状态。

## `State` 类

```python
class State:
    INVALID = 0
    IDLE = 1        # 闲置/阻尼模式
    RL_INIT = 2     # 初始化过渡模式（缓慢移动到初始姿态）
    RL_RUNNING = 3  # RL 策略接管控制模式
```

在后面的 `Humanoid` 类的 `step` 方法中的 `match-case` 结构中，根据当前状态执行不同的控制逻辑：

```python
match (self.state):
    # 状态 1: 闲置/准备
    case State.IDLE:
        # 目标设为当前测量值 -> 也就是不发力移动
        self.joint_position_target[:] = self.joint_position_measured[:]
        # 检测手柄是否按下切换键 -> 进入初始化
        if self.next_state == State.RL_INIT:
            # ... 切换电机到位置控制模式 (Position Mode) ...

    # 状态 2: 初始化 (站起来)
    case State.RL_INIT:
        # 线性插值：从当前趴下的姿态，在 100 步内平滑过渡到站立姿态
        if self.init_percentage < 1.0:
            self.init_percentage += 1 / 100.0
            self.joint_position_target = linear_interpolate(self.starting_positions, self.rl_init_positions, self.init_percentage)
        else:
            # 站好后，等待手柄切换到 RUNNING
            if self.next_state == State.RL_RUNNING:
                self.state = self.next_state

    # 状态 3: RL 运行中
    case State.RL_RUNNING:
        # 直接将神经网络输出的 actions 作为目标位置
        for i in range(len(self.joints)):
            self.joint_position_target[i] = actions[i]
```

## `linear_interpolate` 函数

`linear_interpolate` 函数用于在初始化阶段平滑过渡机器人姿态，从趴下到站立。它接受起始位置、目标位置和当前的过渡百分比，返回一个新的目标位置。

```python
def linear_interpolate(start: np.ndarray, end: np.ndarray, percentage: float) -> np.ndarray:
    percentage = min(max(percentage, 0.0), 1.0)
    target = start * (1. - percentage) + end * percentage
    return target
```

## `Humanoid` 类

`Humanoid` 类代表真实的机器人实体，它封装了所有硬件通信细节。

### `__init__` 方法

#### 硬件通信链路初始化 (CAN总线)

- 左臂绑定到 `can0`
- 右臂绑定到 `can1`
- 左腿绑定到 `can2`
- 右腿绑定到 `can3`

```python
self.left_arm_transport = recoil.Bus("can0")
self.right_arm_transport = recoil.Bus("can1")
self.left_leg_transport = recoil.Bus("can2")
self.right_leg_transport = recoil.Bus("can3")
```

#### 关节映射列表 self.joints
格式：(总线对象, 电机ID, 关节名称)
左腿 ID 均为奇数 (1, 3, 5, 7, 11, 13)，右腿 ID 均为偶数 (2, 4, 6, 8, 12, 14)
每条腿 6 个自由度：跨部(Roll/Yaw/Pitch)、膝盖(Pitch)、脚踝(Pitch/Roll)

```python
self.joints = [
    # 左臂 (1-5)
    (self.left_arm_transport,   1,  "left_shoulder_pitch_joint"),
    (self.left_arm_transport,   3,  "left_shoulder_roll_joint"),
    (self.left_arm_transport,   5,  "left_shoulder_yaw_joint"),
    (self.left_arm_transport,   7,  "left_elbow_pitch_joint"),
    (self.left_arm_transport,   9,  "left_wrist_yaw_joint"),
    # 右臂 (6-10)
    (self.right_arm_transport,  2,  "right_shoulder_pitch_joint"),
    (self.right_arm_transport,  4,  "right_shoulder_roll_joint"),
    (self.right_arm_transport,  6,  "right_shoulder_yaw_joint"),
    (self.right_arm_transport,  8,  "right_elbow_pitch_joint"),
    (self.right_arm_transport,  10, "right_wrist_yaw_joint"),
    # 左腿 (11-16)
    (self.left_leg_transport,   1,  "left_hip_roll_joint"),
    (self.left_leg_transport,   3,  "left_hip_yaw_joint"),
    (self.left_leg_transport,   5,  "left_hip_pitch_joint"),
    (self.left_leg_transport,   7,  "left_knee_pitch_joint"),
    (self.left_leg_transport,   11, "left_ankle_pitch_joint"),
    (self.left_leg_transport,   13, "left_ankle_roll_joint"),
    # 右腿 (17-22)
    (self.right_leg_transport,  2,  "right_hip_roll_joint"),
    (self.right_leg_transport,  4,  "right_hip_yaw_joint"),
    (self.right_leg_transport,  6,  "right_hip_pitch_joint"),
    (self.right_leg_transport,  8,  "right_knee_pitch_joint"),
    (self.right_leg_transport,  12, "right_ankle_pitch_joint"),
    (self.right_leg_transport,  14, "right_ankle_roll_joint"),
]
```

#### 惯性测量单元（IMU）初始化

```python
self.imu = SerialImu(baudrate=Baudrate.BAUD_460800)
self.imu.run_forever()
```

#### 遥控手柄（Joystick/Gamepad）初始化

```python
# Start joystick thread
self.command_controller = Se2Gamepad()
self.command_controller.run()
```

#### 系统状态机初始化

```python
self.state = State.IDLE
self.next_state = State.IDLE
```

- `self.state`：当前状态
- `self.next_state`：下一个状态

#### 关键参数更新

```python
# 3. 更新 RL 初始位置 (长度必须为 22)
# 前 10 个是手臂（通常设为 0），后 12 个是腿部
self.rl_init_positions = np.zeros(22, dtype=np.float32)
self.rl_init_positions[10:] = [0.0, 0.0, -0.2, 0.4, -0.3, 0.0, 0.0, 0.0, -0.2, 0.4, -0.3, 0.0]

# 4. 更新轴向修正 (长度必须为 22)
self.joint_axis_directions = np.ones(22, dtype=np.float32)
# 这里需要根据手臂电机的实际安装方向修改前 10 位
self.joint_axis_directions[10:] = [-1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1]

# 5. 更新观测缓冲区维度
# 4(四元数) + 3(角速度) + 22(关节位置) + 22(关节速度) + 1(模式) + 3(指令) = 55
self.n_lowlevel_states = 4 + 3 + 22 + 22 + 1 + 3
self.lowlevel_states = np.zeros(self.n_lowlevel_states, dtype=np.float32)
```

#### RL 初始化控制器变量

用于平滑过渡（Interpolation）

```python
# 初始化进度（0.0 到 1.0）
self.init_percentage = 0.0
# 用于存储启动初始化那一刻的关节位置
self.starting_positions = np.zeros_like(self.joint_position_target, dtype=np.float32)
```

#### 加载硬件校准文件

在实验室组装机器人时，电机的编码器零点和物理上的“腿部垂直”位置通常会有几度的偏差。这些偏差需要通过校准文件进行补偿。

```python
config_path = "calibration.yaml"
with open(config_path, "r") as f:
    config = OmegaConf.load(f) # 使用 OmegaConf 加载 YAML 配置
position_offsets = np.array(config.get("position_offsets", None))
```

#### 安全性检查与赋值

```python   
# 强制检查：确保校准文件中的偏移量数量与实际电机数量一致（12个或22个）
assert position_offsets.shape[0] == len(self.joints)
# 将校准值存入类的属性中，供后续 update_joints 函数使用
self.position_offsets[:] = position_offsets
```

### `enter_damping` 方法

进入阻尼模式（Damping Mode）

在阻尼模式下，电机不会主动旋转到某个角度，但会像“液压杆”一样产生阻力，防止机器人因为重力瞬间瘫痪倒地，同时也保护电机不被突发的巨大力矩烧毁。

#### 初始化参数数组，长度与电机数量一致

```python
self.joint_kp = np.zeros((len(self.joints),), dtype=np.float32)      # 比例增益（刚度）
self.joint_kd = np.zeros((len(self.joints),), dtype=np.float32)      # 微分增益（阻尼）
self.torque_limit = np.zeros((len(self.joints),), dtype=np.float32)  # 力矩限制
```

#### 设定阻尼模式下的安全参数

```python
# Kp=20: 较低的刚度，电机不会剧烈反弹
# Kd=2:  一定的阻尼，使动作变得粘稠、平滑
# Torque Limit=4: 较低的力矩上限（4Nm），即使发生碰撞也不会伤人或损坏结构
self.joint_kp[:] = 20
self.joint_kd[:] = 2
self.torque_limit[:] = 4
```

#### 遍历每一个关节，逐个通过 CAN 总线发送配置

```python
for i, entry in enumerate(self.joints):
    # 解包关节信息：总线接口、电机ID、关节名称
    bus, device_id, joint_name = entry

    print(f"Initializing joint {joint_name}:")
    print(f"  kp: {self.joint_kp[i]}, kd: {self.joint_kd[i]}, torque limit: {self.torque_limit[i]}")
    
    # 首先将模式设为 IDLE（空闲），停止电机当前的任何动作
    bus.set_mode(device_id, recoil.Mode.IDLE)
    # 必须休眠 1ms，给 CAN 总线和电机驱动器处理指令的时间
    time.sleep(0.001)
    
    # 写入位置环比例增益 Kp
    bus.write_position_kp(device_id, self.joint_kp[i])
    time.sleep(0.001)
    
    # 写入位置环微分增益 Kd
    bus.write_position_kd(device_id, self.joint_kd[i])
    time.sleep(0.001)
    
    # 写入力矩限制，这是防止过载的“保险丝”
    bus.write_torque_limit(device_id, self.torque_limit[i])
    time.sleep(0.001)
    
    # “喂狗”操作：发送心跳包，告诉电机驱动器通信正常，不要进入保护模式
    bus.feed(device_id)
    
    # 最后，正式切换到 DAMPING（阻尼）模式
    # 此时电机开始受电，你会听到轻微电流声，感觉到关节变“硬”了
    bus.set_mode(device_id, recoil.Mode.DAMPING)
```

`bus` 是 `recoil.Bus` 对象（在`Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/recoil/core.py`中定义），表示与电机通信的 `CAN` 总线接口

### `stop` 方法

#### 关闭异步传感器线程

```python
# 停止 IMU 串口读取线程
self.imu.stop()
# 停止手柄输入监听线程
self.command_controller.stop()
```

#### 进入阻尼模式

```python
for entry in self.joints:
    bus, device_id, _ = entry
    # 将所有电机切换到 DAMPING（阻尼）模式
    # 作用：此时电机虽然不再主动发力，但会产生阻力，像“软刹车”一样
    # 防止机器人因为重力瞬间“瘫痪”倒地，能缓慢蹲下或维持姿势
    bus.set_mode(device_id, recoil.Mode.DAMPING)
```

#### 阻塞等待

```python
try:
    # 进入死循环，此时机器人处于有阻力的“挂起”状态
    # 这样操作员有时间扶住机器人，或者将其放回架子上
    while True:
        pass
except KeyboardInterrupt:
    # 当用户第二次按下 Ctrl+C 时，触发最终关机
    print("Exiting damping mode.")
```

#### 彻底断电

```python
for entry in self.joints:
    bus, device_id, _ = entry
    # 将电机设为 IDLE（空闲/完全断电）模式
    # 此时电机完全失去磁力，关节变得彻底松弛（Limp）
    bus.set_mode(device_id, recoil.Mode.IDLE)
```

#### 清理底层通信接口

`*_transport.stop()` 是为了关闭 Linux 系统的 `socketcan` 接口

```python
# 如果有手臂，关闭手臂的 CAN 总线（当前注释中）
self.left_arm_transport.stop()
self.right_arm_transport.stop()

# 关闭左右腿的 CAN 总线，释放 SocketCAN 资源
self.left_leg_transport.stop()
self.right_leg_transport.stop()
```

### `get_observations` 方法

#### 内存视图切片

```python
imu_quaternion = self.lowlevel_states[0:4]       # 0-3: 四元数 (w, x, y, z)
imu_angular_velocity = self.lowlevel_states[4:7] # 4-6: 角速度
joint_positions = self.lowlevel_states[7:19]     # 7-18: 12个关节的角度
joint_velocities = self.lowlevel_states[19:31]   # 19-30: 12个关节的速度
mode = self.lowlevel_states[31:32]               # 31: 当前运行模式
velocity_commands = self.lowlevel_states[32:35]  # 32-34: 目标线速度x, y 和角速度yaw
```

#### 更新传感器数据

它的核心任务是将来自不同硬件（IMU、电机、手柄）的离散数据，封装成一个符合神经网络输入要求的 ** 35 维观测向量**

```python
# 从 IMU 线程读取最新的四元数数据
imu_quaternion[:] = self.imu.quaternion[:]

# 【关键】单位转换：IMU 返回的是度/秒 (deg/s)，但 RL 算法期望弧度/秒 (rad/s)
# 必须在此转换，否则机器人会因为感知的旋转速度过大而疯狂抽搐
imu_angular_velocity[:] = np.deg2rad(self.imu.angular_velocity[:])

# 将底层 CAN 总线读取到的电机实际位置和速度填入缓冲区
joint_positions[:] = self.joint_position_measured[:]
joint_velocities[:] = self.joint_velocity_measured[:]
```

#### 更新用户指令

```python
# 读取手柄上的模式开关（如：切换到 3.0 代表开启 RL 行走）
mode[0] = self.command_controller.commands["mode_switch"]

# 读取摇杆给出的目标速度
velocity_commands[0] = self.command_controller.commands["velocity_x"]
velocity_commands[1] = self.command_controller.commands["velocity_y"]
velocity_commands[2] = self.command_controller.commands["velocity_yaw"]

# 将手柄的模式指令同步到系统状态机的“下一状态”变量中
self.next_state = self.command_controller.commands["mode_switch"]

# 返回填充完毕的完整 35 维向量
return self.lowlevel_states
```

### `update_joint_group` 方法

#### 从算法到硬件 (Sim -> Real)

##### 计算相对位置

```python
position_target_l = (self.joint_position_target[joint_id_l] + self.position_offsets[joint_id_l]) * self.joint_axis_directions[joint_id_l]
position_target_r = (self.joint_position_target[joint_id_r] + self.position_offsets[joint_id_r]) * self.joint_axis_directions[joint_id_r]
```

##### 发送 CAN 数据包

```python
# self.joints[id][0] 是 Bus 对象，[id][1] 是电机 ID
# transmit_pdo_2 是一种高效的实时数据帧，同时发送位置目标。
# 这里将 velocity_target 设为 0，意味着主要依靠电机的内部位置环(Kp)来跟踪。
self.joints[joint_id_l][0].transmit_pdo_2(self.joints[joint_id_l][1], position_target=position_target_l, velocity_target=0.0)
self.joints[joint_id_r][0].transmit_pdo_2(self.joints[joint_id_r][1], position_target=position_target_r, velocity_target=0.0)   
```

#### 从硬件到算法 (Real -> Sim)

##### 接收电机反馈

```python
position_measured_l, velocity_measured_l = self.joints[joint_id_l][0].receive_pdo_2(self.joints[joint_id_l][1])
position_measured_r, velocity_measured_r = self.joints[joint_id_r][0].receive_pdo_2(self.joints[joint_id_r][1])
```

##### 逆向转换

```python
if position_measured_l is not None:
    # 逆公式：算法测量位置 = (硬件原始位置 * 轴向系数) - 校准偏移
    self.joint_position_measured[joint_id_l] = (position_measured_l * self.joint_axis_directions[joint_id_l]) - self.position_offsets[joint_id_l]

if velocity_measured_l is not None:
    # 速度只需要修正方向，不需要修正偏置（导数不含常数项）
    self.joint_velocity_measured[joint_id_l] = velocity_measured_l * self.joint_axis_directions[joint_id_l]

# 右腿执行同样的逻辑
if position_measured_r is not None:
    self.joint_position_measured[joint_id_r] = (position_measured_r * self.joint_axis_directions[joint_id_r]) - self.position_offsets[joint_id_r]

if velocity_measured_r is not None:
    self.joint_velocity_measured[joint_id_r] = velocity_measured_r * self.joint_axis_directions[joint_id_r]
```     

### `reset` 方法

重置机器人环境接口。

```python
def reset(self):
    # 1. 调用 get_observations() 获取当前硬件的最底层状态（IMU、关节、手柄指令）
    obs = self.get_observations()
    
    # 2. 返回这个观测向量，作为策略网络推理的起点
    return obs
```

### `step` 方法

该函数通过一个有限状态机 (Finite State Machine) 来管理机器人的行为，确保机器人能够安全地从静止状态过渡到行走状态。

```python
def step(self, actions: np.ndarray):
    """
    执行一个控制周期。
    参数 actions: 神经网络输出的动作向量（目标关节位置）。
    """
    # 使用 match-case 语法处理不同的系统状态
    match (self.state):
        
        # --- 状态 1: IDLE (闲置/待机) ---
        case State.IDLE:
            # 目标位置始终设为当前测量值，确保电机不发力移动，保持“瘫软”或“阻尼”状态
            self.joint_position_target[:] = self.joint_position_measured[:]

            # 检查手柄是否下达了“进入初始化”的指令
            if self.next_state == State.RL_INIT:
                print("Switching to RL initialization mode")
                self.state = self.next_state # 切换状态

                # 遍历所有关节，将其从阻尼模式切换到位置控制模式
                for entry in self.joints:
                    bus, device_id, _ = entry
                    bus.feed(device_id) # 喂狗（激活通信）
                    bus.set_mode(device_id, recoil.Mode.POSITION) # 开启位置环控制

                # 记录切换瞬间的关节位置，作为插值的起点
                self.starting_positions = self.joint_position_target[:]
                self.init_percentage = 0.0 # 重置插值进度

        # --- 状态 2: RL_INIT (初始化过渡/缓慢站立) ---
        case State.RL_INIT:
            print(f"init: {self.init_percentage:.2f}")
            # 如果进度未完成（小于 100%）
            if self.init_percentage < 1.0:
                # 每一帧增加 1% 的进度（如果是 500Hz 频率，则站立过程耗时 0.2 秒）
                self.init_percentage += 1 / 100.0
                self.init_percentage = min(self.init_percentage, 1.0) # 防止溢出

                # 【核心逻辑】线性插值：让关节从初始位置平滑移动到 RL 算法要求的站立姿态
                # 这样可以防止机器人突然“弹起”造成硬件损坏
                self.joint_position_target = linear_interpolate(self.starting_positions, self.rl_init_positions, self.init_percentage)
            else:
                # 初始化完成后，检查手柄是否下达“正式运行”指令
                if self.next_state == State.RL_RUNNING:
                    print("Switching to RL running mode")
                    self.state = self.next_state

                # 如果用户想切回 IDLE，则进入阻尼模式
                if self.next_state == State.IDLE:
                    print("Switching to idle mode")
                    self.state = self.next_state
                    for entry in self.joints:
                        bus, device_id, _ = entry
                        bus.set_mode(device_id, recoil.Mode.DAMPING)

        # --- 状态 3: RL_RUNNING (RL 策略正式接管控制) ---
        case State.RL_RUNNING:
            # 直接将神经网络输出的 actions 映射到电机的目标位置
            for i in range(len(self.joints)):
                self.joint_position_target[i] = actions[i]

            # 检查手柄是否下达“紧急停止/切回待机”指令
            if self.next_state == State.IDLE:
                print("Switching to idle mode")
                self.state = self.next_state
                for entry in self.joints:
                    bus, device_id, _ = entry
                    bus.set_mode(device_id, recoil.Mode.DAMPING)

    # 【执行层】将计算好的 self.joint_position_target 通过 CAN 总线发送给电机
    self.update_joints()

    # 【感知层】采集当前时刻最新的 IMU 和电机数据，封装成观测向量
    obs = self.get_observations()

    # 返回观测值，供给神经网络下一帧推理使用
    return obs
```

### `update_joints` 方法

这段代码最核心的地方在于它将 左腿（索引 0-5） 和 右腿（索引 6-11） 的对应关节进行了“成对更新”。

```python
def update_joints(self):
    """
    全量更新所有关节的通信函数。
    按照 6 组配对，依次处理左腿和右腿对应的电机。
    """
    # 每一行代表一组对称的关节：(左腿关节索引, 右腿关节索引)
    
    # 1. 更新左右髋部侧摆 (Hip Roll)
    self.update_joint_group(0, 6)
    
    # 2. 更新左右髋部偏航/转动 (Hip Yaw)
    self.update_joint_group(1, 7)
    
    # 3. 更新左右髋部俯仰 (Hip Pitch)
    self.update_joint_group(2, 8)
    
    # 4. 更新左右膝盖俯仰 (Knee Pitch)
    self.update_joint_group(3, 9)
    
    # 5. 更新左右脚踝俯仰 (Ankle Pitch)
    self.update_joint_group(4, 10)
    
    # 6. 更新左右脚踝侧摆 (Ankle Roll)
    self.update_joint_group(5, 11)
```

### `check_connection` 方法

遍历电机列表，逐一发送 Ping 指令并等待反馈。

```python
for entry in self.joints:
    # 解包关节信息：总线对象、电机ID、关节名称
    bus, device_id, joint_name = entry
    
    # 打印当前正在检查的关节名称
    # end="\t" 的作用是让结果 (OK/ERROR) 显示在名称后面，而不是换行
    print(f"Pinging {joint_name} ... ", end="\t")
    
    # 【核心操作】调用底层总线接口发送 Ping 帧
    # 它会往 CAN 总线发一个查询包，如果对应 ID 的电机驱动器在线，会回传一个应答包
    result = bus.ping(device_id)
    
    # 判断返回结果
    if result:
        print("OK")    # 通讯正常
    else:
        print("ERROR") # 通讯失败，可能是没通电、CAN线断了或 ID 设置错误
    
    # 每次检查后休眠 0.1 秒
    # 作用：防止短时间内发送过多查询帧导致 CAN 总线拥堵，保证读取的可靠性
    time.sleep(0.1)
```