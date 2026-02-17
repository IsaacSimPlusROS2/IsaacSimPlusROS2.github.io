---
title: Berkeley Humanoid Lite 的 environments/mujoco.py
date: 2026-02-10 19:59:35
tags: [Berkeley Humanoid Lite, Python]
categories: [Berkeley Humanoid Lite]
---

## 辅助函数

### `quat_rotate_inverse(q, v)`

*   **功能**：计算向量 $v$ 经过四元数 $q$ 的**逆旋转**后的结果。
*   **数学含义**：通常用于将 **世界坐标系** 下的向量（如重力向量 $[0, 0, -1]$）转换到 **机器人基座（Base）坐标系** 下。这是人形机器人 RL 中常用的观测特征，用于让机器人感知“哪边是下”。

## 类详解

### `class MujocoEnv` (基类)

这是仿真环境的基础封装。
*   **模型加载**：根据 `cfg.num_joints` 自动判断加载哪种 XML 模型文件：
    *   22关节：全尺寸人形 (`bhl_scene.xml`)。
    *   其他：双足版本 (`bhl_biped_scene.xml`)。
*   **初始化**：
    *   `mj_model`: 静态模型定义。
    *   `mj_data`: 动态仿真数据（位置、速度、力等）。
    *   `mj_viewer`: 启动被动的 3D 查看器。

### `class MujocoVisualizer` (可视化器)

**用途**：用于“回放”或“监视”。它**不进行物理控制**，而是强制修改机器人的状态。
*   **`reset`**: 将机器人重置到原点，姿态归零。
*   **`step(robot_observations)`**:
    *   **输入**：`robot_observations`（全状态观测向量）。
    *   **逻辑**：它**直接覆盖** `mj_data.qpos` (位置) 和 `mj_data.qvel` (速度)。
    *   **意义**：这实现了“运动学回放”。即使物理上不合理（例如穿模），它也会强制显示输入的状态。通常用于将真实机器人的 log 数据在仿真器中显示出来。

### `class MujocoSimulator` (物理仿真器)
**用途**：用于 **Sim-to-Real** 训练或推理。它模拟真实的物理动力学。

**关键属性：**
*   `physics_substeps`：**物理子步数**。
    *   RL 策略频率通常较低（如 50Hz，即 `policy_dt`=0.02s）。
    *   物理引擎频率较高（如 1000Hz，即 `physics_dt`=0.001s）。
    *   `substeps = 20` 表示策略每输出一次动作，物理引擎计算 20 次。
*   `joint_kp` / `joint_kd`：PD 控制器的比例（刚度）和微分（阻尼）增益。
*   `effort_limits`：关节力矩限制（模拟电机的最大输出）。
*   `Se2Gamepad`：启动一个手柄监听线程，用于在仿真中通过手柄控制机器人的移动指令（x速度, y速度, 偏航角速度）。

**关键方法：**

*   **`reset()`**:
    *   将机器人设为 `default_base_position` 和 `default_joint_positions`。
    *   返回初始观测值。

*   **`step(actions)` (核心循环)**:
    1.  **子步循环**：运行 `physics_substeps` 次。
    2.  **`_apply_actions(actions)`**:
        *   **PD 控制律**：
            $$ \tau = K_p (q_{target} - q_{current}) + K_d (0 - \dot{q}_{current}) $$
            *注意：这里目标速度设为了0，是标准的位置控制模式。*
        *   **限幅**：将计算出的力矩 $\tau$ 限制在 `effort_limits` 范围内。
        *   **执行**：写入 `mj_data.ctrl`。
    3.  **物理步进**：`mujoco.mj_step`。
    4.  **实时同步**：计算物理计算消耗的时间，如果有剩余时间，通过 `time.sleep` 等待，以保持仿真画面与现实时间同步（Real-time factor = 1）。
    5.  **返回观测**：调用 `_get_observations`。

*   **`_get_observations()` (构建状态空间)**:
    *   组合机器人状态和用户指令，返回一个 PyTorch Tensor。
    *   **观测向量结构** (Concatenated):
        1.  `Base Quat` (4维): 基座姿态四元数。
        2.  `Base Ang Vel` (3维): 基座角速度。
        3.  `Joint Pos` (N维): 关节角度。
        4.  `Joint Vel` (N维): 关节角速度。
        5.  `Commands` (4维): `[模式, X速度, Y速度, Yaw速度]`。

---

## 代码中的常量与配置映射表

虽然代码中主要引用了外部 `cfg`，但根据 `MujocoSimulator` 的逻辑，可以整理出以下关键参数与数据的关系：

| 参数/变量 | 数据来源 | 描述 |
| :--- | :--- | :--- |
| **Physics Timestep** | `cfg.physics_dt` | 底层物理引擎的积分步长 (通常 0.001s 或 0.002s) |
| **Policy Timestep** | `cfg.policy_dt` | RL 策略的控制周期 (通常 0.02s) |
| **Substeps** | `policy_dt / physics_dt` | 每次 `step()` 调用对应的物理步数 |
| **Gravity** | `[0, 0, -1]` | 重力向量（世界坐标系），用于计算投影重力 |
| **Joint KP** | `cfg.joint_kp` | 关节位置增益 (Stiffness) |
| **Joint KD** | `cfg.joint_kd` | 关节速度增益 (Damping) |
| **Torque Limits** | `cfg.effort_limits` | 关节最大输出力矩 (N·m) |
| **Sensordata Layout** | `mj_data.sensordata` | MuJoCo XML 中定义的传感器数据缓冲区 |

**Observation (观测) 数据结构表:**

| 组成部分 | 维度 | 来源 | 说明 |
| :--- | :--- | :--- | :--- |
| **Base Quat** | 4 | `sensordata` | IMU 测量的姿态 (w, x, y, z) |
| **Base Ang Vel** | 3 | `sensordata` | IMU 测量的角速度 (wx, wy, wz) |
| **Joint Pos** | N | `sensordata` | 编码器读取的关节角度 |
| **Joint Vel** | N | `sensordata` | 编码器读取的关节角速度 |
| **Command** | 4 | `Se2Gamepad` | `[Mode, Vel_X, Vel_Y, Vel_Yaw]` |