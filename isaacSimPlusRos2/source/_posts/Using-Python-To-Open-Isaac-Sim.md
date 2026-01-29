---
title: 使用 Python 打开 Isaac Sim
date: 2026-01-29 18:13:32
tags: [Isaac Sim, Python]
categories: [示例教程]
---

## 前言

要确保 Isaac Sim、Python、ROS 2、VS Code、Miniconda 都已正确安装和配置。

## 安装依赖

```bash
sudo apt install cmake build-essential -y
```

## 安装 Isaac Lab

使用以下命令Clone Isaac Lab：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
```

### 链接Isaac Sim和IsaacLab

```bash
cd IsaacLab
ln -s ${HOME}/isaacsim _isaac_sim
```

将 `isaacsim` 替换为 Isaac Sim 的实际路径。

### 创建并激活Conda环境

```bash
conda create -n isaaclab python=3.10 -y
conda activate isaaclab
```

用 Python 3.10 是为了确保与 Ros 2 Humble 兼容。

### 安装Isaac Lab依赖

```bash
./isaaclab.sh --install
```

## 配置 `.bashrc`

在 `~/.bashrc` 文件中添加以下内容：

```bash
source isaac-sim地址/setup_conda_env.sh
source isaac-sim地址/setup_ros_env.sh
```

## 写出第一个 Python 代码来启动 Isaac Sim

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World

world = World(stage_units_in_meters=1.0)
stage = world.stage

world.reset()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
```

理论上，如果配置成功，你就会看到 Isaac Sim Python 5.1