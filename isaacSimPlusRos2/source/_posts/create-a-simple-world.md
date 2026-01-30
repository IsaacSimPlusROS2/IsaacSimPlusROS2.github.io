---
title: 创建一个简单的世界
date: 2026-01-29 23:12:09
tags: [Isaac Sim, Python]
categories: [初级]
---

## 如何启动 Isaac Sim

在 VS Code 内输入以下代码

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World

world = World(stage_units_in_meters=1.0)

world.reset()
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
```

理论上，如果配置成功，你就会看到 Isaac Sim Python 5.1

## 加载 Ros 2

```python
import omni.kit.app
from isaacsim.core.utils.extensions import enable_extension

# 先启用扩展
enable_extension("omni.graph.action")
enable_extension("omni.syntheticdata")
enable_extension("isaacsim.ros2.bridge")  # 新版扩展名
enable_extension("omni.isaac.ros2_bridge")  # 兼容旧版
enable_extension("omni.replicator.core")    # 用于 render_product

# 让扩展完全加载
for _ in range(10):
    omni.kit.app.get_app().update()
```

## 获得舞台

```python
stage = world.stage
```

## 创建一个默认的地面

```python
from isaacsim.core.utils.prims import create_prim

create_prim("/World/GroundPlane", "Xform")
world.scene.add_default_ground_plane()
```

## 创建灯光

```python
sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
sun.CreateIntensityAttr(2000.0)
sun.CreateAngleAttr(1.0)
```

## 创建一个 Cube

```python
from isaacsim.core.api.objects.cuboid import DynamicCuboid
from pxr import UsdGeom

cube = DynamicCuboid(
    prim_path="/World/Cube",
    name="cube",
    position=(0.6, 0.0, 0.4),
    size=0.08,
)
world.scene.add(cube)
cube_prim = stage.GetPrimAtPath("/World/Cube")
UsdGeom.Gprim(cube_prim).CreateDisplayColorAttr([(1.0, 0.2, 0.2)])
```

> 且看上面的代码  
> 我们在 $x$ 为 0.6, $y$ 为  0.0, $z$ 为 0.4 的地方上面创建了一个大小是 0.08 的 cube

## 创建一个 Franka 机械臂

```python
FRANKA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
)

franka = stage.DefinePrim("/World/FrankaRoot/Franka", "Xform")
franka.GetReferences().AddReference(FRANKA_USD)
franka.Load()
```

> 引入一个在线的 usd 资产，最好使用 `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/` + xxxx
> 如何查看呢？看下图
> ![](/imgs/image4.png)
> 选择 `Copy URL Link` 即可

## 源代码

[点击前往 GitHub 查看](https://github.com/IsaacSimPlusROS2/create-a-simple-world/blob/main/world.py)