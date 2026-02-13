---
title: 弄一个简单的 ROS 2 程序
date: 2026-02-13 11:50:07
tags: [ROS 2, Python]
categories: [ROS 2]
---

## 前提条件

确保你的 Conda 环境中安装了 rclpy 包。如果没有安装，可以使用以下命令进行安装：

```bash
pip install rclpy
```

## 创建 ROS 2 节点

创建一个名为 `simple_ros2_node.py` 的 Python 文件，并添加以下代码：

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('robot_brain')
        # 创建一个发布者，发布到 'chatter' 话题
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        # 每 3 秒发一次
        self.timer = self.create_timer(3, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Robot is walking...'
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main():
    rclpy.init() # 初始化
    node = MyRobotNode()
    rclpy.spin(node) # 开始循环处理
    rclpy.shutdown() # 关闭

if __name__ == '__main__':
    main()
```

## 运行节点

在终端中运行以下命令来启动你的 ROS 2 节点：

```bash
python node_one.py
```

此时，你运行 `ros2 topic list` 会看到 `/chatter`。

![](/imgs/image6.png)

## 监听 `node_one.py` 发出的东西

在终端中运行以下命令：

```bash
ros2 topic echo /chatter
```

你就会看到 `data: Robot is walking...`

## 源码

[IsaacSimPlusROS2/An-Easy-Ros-2-Program](https://github.com/IsaacSimPlusROS2/An-Easy-Ros-2-Program)