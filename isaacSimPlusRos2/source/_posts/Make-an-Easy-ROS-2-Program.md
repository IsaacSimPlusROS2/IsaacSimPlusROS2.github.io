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

## 创建 ROS 2 发布节点

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

## 创建 ROS 2 监听节点

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber') # 初始化节点名称
        
        # 创建订阅者：消息类型 String, 话题名称 'chatter', 回调函数, 队列长度 10
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # 防止变量被垃圾回收 (Python特性)

    def listener_callback(self, msg):
        # 收到消息时的回调函数
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()

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

或者

```bash
python node_two.py 
```

你就会看到 `[INFO] [1770962759.315087388] [minimal_subscriber]: I heard: "Robot is walking..."`

## 源码

[GitHub: IsaacSimPlusROS2/An-Easy-Ros-2-Program](https://github.com/IsaacSimPlusROS2/An-Easy-Ros-2-Program)