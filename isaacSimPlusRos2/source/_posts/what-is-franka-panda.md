---
title: 什么是 FrankaPanda
date: 2026-02-02 22:36:50
tags: [Isaac Sim, FrankaPanda, Franka]
categories: [FrankaPanda]
---

## FrankaPanda 的长相

![FrankaPanda 的长相](/imgs/image5.png)

## 用 Python 代码加载 FrankaPanda

```python
from omni.isaac.franka import Franka

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = world.stage

FRANKA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
)

franka_root = stage.DefinePrim("/World/FrankaRoot", "Xform")
franka_prim = stage.DefinePrim("/World/FrankaRoot/Franka", "Xform")
franka_prim.GetReferences().AddReference(FRANKA_USD)
franka_prim.Load()

# 包装为机器人对象
franka_robot = world.scene.add(
    Franka(
        prim_path="/World/FrankaRoot/Franka", 
        name="my_franka"
    )
)
```

## FrankaPandas 控制器的代码（非 ROS 2）

```python
from omni.isaac.franka.controllers import PickPlaceController

controller = PickPlaceController(
    name="pick_place_controller",
    gripper=franka_robot.gripper,
    robot_articulation=franka_robot
)
```

## FrankaPandas 的基础动作

### 引入动作模块

```python
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
```

### 爪子控制

#### 张开动作

```python
action = franka_robot.gripper.forward(action="open")
franka_robot.apply_action(action)   # 应用动作
```

#### 闭合爪子 (抓取)

```python
action = franka_robot.gripper.forward(action="close")
franka_robot.apply_action(action)   # 应用动作
```