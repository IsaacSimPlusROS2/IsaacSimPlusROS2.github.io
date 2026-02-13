---
title: 连接到在另一台计算机上运行的Subscriber
date: 2026-02-13 18:46:23
tags: [ROS 2, Ubuntu]
categories: [ROS 2]
---

## 何为“另一台计算机”

你可以认为是处于同一局域网内的另一台计算机，或者是通过VPN连接的远程计算机。

譬如，发布者计算机的 Ip 地址是 `192.168.0.3`，监听者计算机的 Ip 地址是 `192.168.0.103`.

## 实现办法

### 设置 `ROS_DOMAIN_ID`

ROS 2 使用 `ROS_DOMAIN_ID` 来隔离网络中的不同机器人群组。

所有电脑必须使用相同的 Domain ID。

默认 ID 是 `0`。

我们可以在发布者和监听者的电脑上设置相同的 `ROS_DOMAIN_ID`，例如：

```bash
export ROS_DOMAIN_ID=30
```

即将 `ROS_DOMAIN_ID` 设置为 `30`，当然，你也可以选择其他数字，只要确保发布者和监听者使用相同的值即可。

### 设置 `ROS_LOCALHOST_ONLY`

`ROS_LOCALHOST_ONLY` 就是“仅本机模式”，如果设置为 `1`，ROS 2 将只在本地计算机上进行通信，不会与其他计算机进行通信。

因此，在发布者和监听者的电脑上都需要将 `ROS_LOCALHOST_ONLY` 设置为 `0`，以允许跨计算机通信：

```bash
export ROS_LOCALHOST_ONLY=0
```

### 暂时关闭防火墙

DDS 使用 UDP 协议进行大量通信。

Linux 的防火墙（如 `ufw` 或 `iptables`）经常会拦截这些包。

因此，在发布者和监听者的电脑上暂时关闭防火墙，以确保 DDS 通信畅通：

```bash
sudo ufw disable
```

### 运行 `node_one.py`

在发布者计算机上运行 `node_one.py`：

```bash
python3 node_one.py
```

### 在监听者计算机上检查时候能否看到发布者的主题

在监听者计算机上运行以下命令，查看是否能够看到发布者的主题：

```bash
ros2 topic list
```

如果一切设置正确，你应该能够看到发布者的主题，例如 `/chatter`。

此时，你可以在监听者计算机上运行 `ros2 topic echo /chatter` 来查看发布者发送的消息：

```bash
ros2 topic echo /chatter
```

如果`topic echo`能够显示发布者发送的消息，那么你已经成功连接到在另一台计算机上运行的Subscriber了。

## 源码

[GitHub: IsaacSimPlusROS2/An-Easy-Ros-2-Program](https://github.com/IsaacSimPlusROS2/An-Easy-Ros-2-Program)