---
title: Berkeley Humanoid Lite 的 scripts/sim2real/visualize.py
date: 2026-02-11 11:31:23
tags: [Berkeley Humanoid Lite, Python]
categories: [Berkeley Humanoid Lite]
---

`scripts/sim2real/visualize.py` 只负责接收数据并画图，不负责控制逻辑。

## 初始化环境

```python
visualizer = MujocoVisualizer(Cfg(
    {
        "num_joints": 12,   # 机器人关节数量（Lite版本通常是12个自由度）
        "physics_dt": 0.001, # 物理仿真步长 1ms
    })
)
```

这里的 `num_joints` 会对应到 `MujocoEnv` 的这个部分

```python
if cfg.num_joints == 22:
    self.mj_model = mujoco.MjModel.from_xml_path("source/berkeley_humanoid_lite_assets/data/mjcf/bhl_scene.xml")
else:
    self.mj_model = mujoco.MjModel.from_xml_path("source/berkeley_humanoid_lite_assets/data/mjcf/bhl_biped_scene.xml")
```

## UDP 接收线程

```python
    def receive_udp_data(robot_observation_buffer):
        # Setup UDP communication
        # 监听 0.0.0.0:11000 (接收所有网卡数据)，目标地址设为本地(发送时用，这里主要只收)
        udp = UDP(("0.0.0.0", 11000), ("127.0.0.1", 11000))

        """Thread function to receive UDP data."""
        while True:
            # 阻塞式接收 Numpy 数组数据
            robot_observations = udp.recv_numpy(dtype=np.float32)
            if robot_observations is not None:
                # [关键] 原地更新缓冲区，而不是创建新变量
                robot_observation_buffer[:] = robot_observations
```

## 共享数据缓冲区

```python
robot_observation_buffer = np.zeros((35,), dtype=np.float32)
```

35 个维度通常包含：
- 基座姿态（四元数 4）
- 基座角速度（3）
- 关节位置（12）
- 关节速度（12）
- 其他（如上一次动作指，或足端接触力等，共 4 个）

具体含义取决于 `berkeley_humanoid_lite` 的定义，但 35 是非常典型的大小。

## 启动线程与主循环

```python
    udp_receive_thread = threading.Thread(target=receive_udp_data, args=(robot_observation_buffer,))
    udp_receive_thread.daemon = True  # 设为守护线程，主程序退出时它也会自动退出
    udp_receive_thread.start()

    while True:
        visualizer.step(robot_observation_buffer)
```

`daemon = True`：防止程序关闭时，后台 UDP 线程还在空转导致进程无法结束。

`visualizer.step` 会根据缓冲区里的最新数据（由 UDP 线程不断刷新），更新 MuJoCo 中机器人的关节角度和基座位置，并渲染一帧画面。

