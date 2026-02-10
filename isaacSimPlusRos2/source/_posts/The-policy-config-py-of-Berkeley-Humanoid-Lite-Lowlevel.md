---
title: Berkeley Humanoid Lite Lowlevel 的 policy/config.py
date: 2026-02-10 23:15:25
tags: [Berkeley Humanoid Lite, Lowlevel, Python]
categories: [Berkeley Humanoid Lite]
---

## Cfg 类

### 策略配置 (`Policy`)

```python
policy_checkpoint_path: str
```

`policy_checkpoint_path`: 训练好的神经网络模型文件（`.pt` 或 `.jit`）的路径。

对应的 `Berkeley-Humanoid-Lite/configs/policy_biped_50hz.yaml` 文件中，它有这样的参数：

```yaml
policy_checkpoint_path: "checkpoints/policy_biped_50hz.onnx"
```

### 网络配置 (`Networking`)

```python
ip_robot_addr: str
ip_policy_obs_port: int
ip_host_addr: str
ip_policy_acs_port: int
```

定义了机器人（Robot）和策略服务器（Host）之间的 IP 地址及端口。这通常用于**离机控制**：算力强大的 PC 跑策略，通过网络将指令发给算力较弱的机器人控制器。

对应的 `Berkeley-Humanoid-Lite/configs/policy_biped_50hz.yaml` 文件中，它有这样的参数：

```yaml
ip_robot_addr: 127.0.0.1
ip_policy_obs_port: 10000
ip_host_addr: 127.0.0.1
ip_policy_acs_port: 10001
```

### 物理/时间步长 (`Physics`)

```python
control_dt: float
policy_dt: float
physics_dt: float
cutoff_freq: float
```

`physics_dt`: 物理引擎计算的步长（如 1ms）。

`control_dt`: 底层 PD 控制器的更新步长。

`policy_dt`: 神经网络策略的决策步长（如 20ms / 50Hz）。

对应的 `Berkeley-Humanoid-Lite/configs/policy_biped_50hz.yaml` 文件中，它有这样的参数：

```yaml
control_dt: 0.004
policy_dt: 0.02
physics_dt: 0.0005
cutoff_freq: 1000
```

### 关节/控制参数 (`Articulation`)

```python
num_joints: int
joints: list[str]
joint_kp: list[float] | float
joint_kd: list[float] | float
effort_limits: list[float]
default_base_position: list[float]
default_joint_positions: list[float]
```

`num_joints`: 关节数量（双足版通常为 12）。

`joint_kp` / `joint_kd`: 底层电机的刚度和阻尼增益。

`effort_limits`: 电机最大扭矩限制。

`default_...`: 机器人重置时的初始位置。

对应的 `Berkeley-Humanoid-Lite/configs/policy_biped_50hz.yaml` 文件中，它有这样的参数：

```yaml
num_joints: 12
joints:
- leg_left_hip_roll_joint
- leg_left_hip_yaw_joint
- leg_left_hip_pitch_joint
- leg_left_knee_pitch_joint
- leg_left_ankle_pitch_joint
- leg_left_ankle_roll_joint
- leg_right_hip_roll_joint
- leg_right_hip_yaw_joint
- leg_right_hip_pitch_joint
- leg_right_knee_pitch_joint
- leg_right_ankle_pitch_joint
- leg_right_ankle_roll_joint
joint_kp:
- 20.0
- 20.0
- 20.0
- 20.0
- 20.0
- 20.0
- 20.0
- 20.0
- 20.0
- 20.0
- 20.0
- 20.0
joint_kd:
- 2.0
- 2.0
- 2.0
- 2.0
- 2.0
- 2.0
- 2.0
- 2.0
- 2.0
- 2.0
- 2.0
- 2.0
effort_limits:
- 5.0
- 5.0
- 5.0
- 5.0
- 5.0
- 5.0
- 5.0
- 5.0
- 5.0
- 5.0
- 5.0
- 5.0
default_base_position:
- 0.0
- 0.0
- 0.0
default_joint_positions:
- 0.0
- 0.0
- -0.2
- 0.4
- -0.3
- 0.0
- 0.0
- 0.0
- -0.2
- 0.4
- -0.3
- 0.0
```

### 观测与历史 (`Observation`)

```python
num_observations: int
history_length: int
```

`num_observations` (观测值维度): 这个数值通常是 35

- 基座姿态 (4维)：四元数 [w, x, y, z]，告诉机器人它是站着的还是歪的。
- 基座角速度 (3维)：陀螺仪数据 [wx, wy, wz]，告诉机器人旋转有多快。
- 关节位置 (12维)：12个电机的当前转角，告诉机器人现在的姿势。
- 关节速度 (12维)：12个电机的转动速度，告诉机器人四肢运动的趋势。
- 控制指令 (4维)：[mode, vx, vy, vyaw]，告诉机器人用户想让它干什么。

`history_length`: 观测序列的长度。人形机器人通常需要过去几帧的数据（History）来弥补传感器延迟或估计线速度。

- 通常在 5 到 20 之间。
- 输入总规模：神经网络实际接收的向量长度 = num_observations × history_length。

### 命令配置（`Command configurations`）

```python
command_velocity: list[float]
```

这个列表通常包含 3 个维度，代表机器人在二维平面上的运动目标。

- **索引 0 (Velocity X):** 前进/后退的目标线速度（单位：米/秒，$m/s$）。
    - 正值代表向前走，负值代表向后走。
- **索引 1 (Velocity Y):** 左右横移的目标线速度（单位：米/秒，$m/s$）。
    - 正值代表向左横移，负值代表向右横移。
- **索引 2 (Yaw Rate):** 转向的目标角速度（单位：弧度/秒，$rad/s$）。
    - 正值代表左转（逆时针），负值代表右转（顺时针）。

在配置文件（YAML）中的样子

对应的 `Berkeley-Humanoid-Lite/configs/policy_biped_50hz.yaml` 文件中，它有这样的参数：

```yaml
command_velocity:
- -0.419607937335968
- -0.05113796889781952
- -0.678446888923645
```

### 动作配置 (`Action`)

```python
num_actions: int
action_indices: list[int]
action_scale: float
action_limit_lower: float
action_limit_upper: float
```

`num_actions`（动作维度）: 神经网络输出层的神经元数量。
- 对于 Berkeley Humanoid Lite 的双足版本，这个值通常是 12（每条腿 6 个关节：髋部转动、髋部侧摆、髋部俯仰、膝盖俯仰、脚踝俯仰、脚踝侧摆）。

`action_indices`（动作索引映射）: 将神经网络输出的 12 个值映射到仿真器或硬件中特定关节的索引。
- 仿真模型（URDF/MJCF）中可能包含 20 多个关节（包括固定不动的躯干、手臂等），但 RL 策略可能只控制其中的 12 个行走关节。
- 通过 `action_indices`，代码知道：`actions[0]` 应该发给 ID 为 `0` 的左跨关节，`actions[1]` 发给 ID 为 `1` 的关节。

`action_scale`（动作缩放因子）: 将神经网络的输出值（通常是归一化的）转化为实际的物理角度（弧度）。
- 神经网络为了训练稳定，其输出通常被限制在 $[-1, 1]$ 之间，但是机器人关节可能需要移动 $\pm 0.5$ 弧度。
- **计算公式**：$\text{实际目标角度} = \text{默认姿态角度} + (Action × action\_scale)$。

`action_limit_lower` / `action_limit_upper`（动作限制）: 对神经网络的原始输出值进行硬性截断（Clip）。
- 确保神经网络不会输出一个过大的值导致电机转动到机械死角或打坏电线。
- 在训练初期，随机的动作可能非常狂野，通过限制动作范围，可以防止机器人因为动作过载而频繁“炸机”。

对应的 `Berkeley-Humanoid-Lite/configs/policy_biped_50hz.yaml` 文件中，它有这样的参数：

```yaml
num_actions: 12
action_scale: 0.25
action_indices:
- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
action_limit_lower: -10000
action_limit_upper: 10000
```