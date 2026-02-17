---
title: Berkeley Humanoid Lite Lowlevel 的 robot/bimanual.py
date: 2026-02-10 20:36:36
tags: [Berkeley Humanoid Lite, Python]
categories: [Berkeley Humanoid Lite]
---

## 代码概述

1.  **硬件拓扑结构**:
    *   **左臂**: 连接在 `can0` 总线上，关节 ID 为奇数 (1, 3, 5, 7, 9)。
    *   **右臂**: 连接在 `can1` 总线上，关节 ID 为偶数 (2, 4, 6, 8, 10)。
    *   **夹爪**: 通过 USB 串口 `/dev/ttyUSB0` 控制。

2.  **运动学修正 (`joint_axis_directions` & `position_offsets`)**:
    *   由于电机安装方向不同，物理旋转方向可能与代码逻辑方向相反。代码中使用 `+1` 或 `-1` 数组来统一坐标系。
    *   **校准逻辑**: 在 `start()` 方法中，机器人将当前的上电位置作为“零点”记录到 `position_offsets` 中。这意味着机器人每次启动时保持的姿势被视为初始位置。

3.  **控制循环 (`step` & `update_joints`)**:
    *   这是一个典型的 **Action -> Observation** 循环。
    *   `step(actions)`: 接收目标角度，更新内部状态。
    *   `update_joints()`:
        *   将目标角度转换为电机指令（应用方向和偏移）。
        *   通过 CAN 总线发送 **PDO (Process Data Object)** 实时控制包。
        *   读取电机的反馈（位置和速度）。
        *   通过串口发送夹爪的 PWM/位置信号。

4.  **安全机制**:
    *   `start()`: 设置力矩限制（Torque Limit），防止电机过载。
    *   `stop()`: 退出时进入 **DAMPING (阻尼)** 模式。这比直接断电更安全，因为电机产生的反电动势会阻碍手臂自由下坠，提供一种“软着陆”效果。

## 配置常量表 (Instance Configuration)

虽然这个类没有显式的 `Class Constants`（静态常量），但在 `__init__` 中定义了固定的硬件映射配置。以下是这些硬编码配置的表格形式：

### 关节映射表 (`self.joints`)

| 索引 | 所在总线 | 设备ID | 关节名称 | 物理位置 |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `can0` (Left) | **1** | `left_shoulder_pitch` | 左肩-俯仰 |
| 1 | `can0` (Left) | **3** | `left_shoulder_roll` | 左肩-横滚 |
| 2 | `can0` (Left) | **5** | `left_shoulder_yaw` | 左肩-偏航 |
| 3 | `can0` (Left) | **7** | `left_elbow_joint` | 左肘 |
| 4 | `can0` (Left) | **9** | `left_wrist_yaw_joint` | 左腕 |
| 5 | `can1` (Right)| **2** | `right_shoulder_pitch` | 右肩-俯仰 |
| 6 | `can1` (Right)| **4** | `right_shoulder_roll` | 右肩-横滚 |
| 7 | `can1` (Right)| **6** | `right_shoulder_yaw` | 右肩-偏航 |
| 8 | `can1` (Right)| **8** | `right_elbow_joint` | 右肘 |
| 9 | `can1` (Right)| **10** | `right_wrist_yaw_joint`| 右腕 |

### 方向修正表 (`self.joint_axis_directions`)

用于将软件坐标系映射到硬件旋转方向。

| 索引 | 关节 | 方向系数 | 含义 |
| :--- | :--- | :--- | :--- |
| 0 | 左肩 Pitch | **+1** | 同向 |
| 1 | 左肩 Roll | **+1** | 同向 |
| 2 | 左肩 Yaw | **-1** | **反向** |
| 3 | 左肘 | **-1** | **反向** |
| 4 | 左腕 | **-1** | **反向** |
| 5 | 右肩 Pitch | **-1** | **反向** |
| 6 | 右肩 Roll | **+1** | 同向 |
| 7 | 右肩 Yaw | **-1** | **反向** |
| 8 | 右肘 | **+1** | 同向 |
| 9 | 右腕 | **-1** | **反向** |
| 10 | 夹爪 (左指) | **+1** | 同向 |
| 11 | 夹爪 (右指) | **+1** | 同向 |

---

## Class `Bimanual` 方法列表

| 方法名 | 参数 | 描述 |
| :--- | :--- | :--- |
| `__init__` | - | 初始化 CAN 总线、串口，并建立关节 ID 映射和方向数组。 |
| `start` | `kp`(刚度), `kd`(阻尼), `torque_limit`(力矩限制) | **启动机器人**。<br>1. 设置 PID 参数和安全限制。<br>2. 读取当前位置作为初始零点 (`position_offsets`)。<br>3. 激活电机进入 `POSITION` 模式。 |
| `stop` | - | **安全停机**。<br>1. 将所有关节设为 `DAMPING` (阻尼) 模式以防止重力坠落。<br>2. 等待用户 Ctrl+C。<br>3. 设为 `IDLE` 并关闭总线。 |
| `get_observations` | - | **获取状态**。<br>返回一个包含 12 个元素的数组：<br>`[10个关节位置, 左夹爪目标, 右夹爪目标]`。 |
| `step` | `actions` (np.ndarray) | **控制步进** (核心控制接口)。<br>输入动作向量（10个关节角度 + 2个夹爪指令），执行硬件通信，并返回最新的观测数据 (`obs`)。 |
| `update_joints` | - | **硬件通信循环**。<br>遍历所有关节对，调用 `update_joint_group` 发送指令并接收反馈。同时通过串口发送夹爪指令。 |
| `update_joint_group`| `joint_id_l`, `joint_id_r` | **底层原子操作**。<br>同时更新左臂和右臂的一对关节（例如左肘和右肘）。<br>处理坐标变换：`目标 = (输入 + 偏移) * 方向`。 |
| `reset` | - | **重置环境**。<br>当前代码实现较简单，仅返回当前的观测值。通常用于 RL 环境的初始化。 |
| `check_connection` | - | **硬件自检**。<br>向所有 10 个关节发送 Ping 指令，检测 CAN 通信是否正常。 |