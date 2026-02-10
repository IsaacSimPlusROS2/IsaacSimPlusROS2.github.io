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

### 网络配置 (`Networking`)

```python
ip_robot_addr: str
ip_policy_obs_port: int
ip_host_addr: str
ip_policy_acs_port: int
```

定义了机器人（Robot）和策略服务器（Host）之间的 IP 地址及端口。这通常用于**离机控制**：算力强大的 PC 跑策略，通过网络将指令发给算力较弱的机器人控制器。

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

*   **索引 0 (Velocity X):** 前进/后退的目标线速度（单位：米/秒，$m/s$）。
    *   正值代表向前走，负值代表向后走。
*   **索引 1 (Velocity Y):** 左右横移的目标线速度（单位：米/秒，$m/s$）。
    *   正值代表向左横移，负值代表向右横移。
*   **索引 2 (Yaw Rate):** 转向的目标角速度（单位：弧度/秒，$rad/s$）。
    *   正值代表左转（逆时针），负值代表右转（顺时针）。

### 动作配置 (`Action`)
