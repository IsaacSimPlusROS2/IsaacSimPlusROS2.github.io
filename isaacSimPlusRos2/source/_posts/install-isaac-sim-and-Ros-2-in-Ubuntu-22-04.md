---
title: 在 Ubuntu 上安装 Isaac Sim 和 ROS 2 Humble
date: 2026-01-29 13:46:25
tags: [Isaac Sim, ROS 2, Ubuntu]
categories: [安装教程]
---

## Ubuntu的选择

推荐使用Ubuntu 22.04.5 desktop amd64

下面是安装教程

[如何安装 Ubuntu 22.04 LTS 桌面版 (图文教程) ? - 知乎](https://zhuanlan.zhihu.com/p/569347838)

对于分区，这部分非常重要，如果分错了，基本上就要重新安装系统了，因为我试过Ext 4在傲梅分区助手扩容的办法，最终，还是失败了。

有个事情，得知道，反复将存储设备格式化，存储设备会报废的（曾经的教训）

为了不出问题，请看下表<sup><a href="#footnote-1" id="ref-1">1</a></sup>，进行分区

| 分区名 | 分区类型 | 分区位置 | 文件格式/用于 | 挂载地址 | 大小（MB） |
| -------- | ---------- | ---------- | ------------------ | ---------- | ------------ |
| EFI 分区 | 主分区 | 空间起始位置 | EFI 系统分区 | /boot/efi（安装时自动） | 2048 |
| SWAP 交换分区 | 主分区 | 空间起始位置 | 交换空间（swap） | —— | 30720 |
| root 文件系统 | 逻辑分区 | 空间起始位置 | Ext4 日志文件系统 | / | 153600 |
| 用户文件系统 | 逻辑分区 | 空间起始位置 | Ext4 日志文件系统 | /home | 剩余全部 |

分区大小的时候，可使用这个进行换算  [GB到MB换算](https://www.bchrt.com/data-storage/gb-to-mb.htm)

如果虚拟机找不到U盘，[ubuntu虚拟机识别不到移动硬盘的一种可能与解决方法_ubuntu读不到硬盘-CSDN博客](https://blog.csdn.net/qq1016019583/article/details/141345771)

## 安装ROS 2

[【ROS2实战】在中国地区 Ubuntu 22.04 上安装 ROS 2 Humble 教程](https://zhuanlan.zhihu.com/p/1905555826527174743)

## 安装 Isaac Sim

[Download Isaac Sim — Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)

下载 [Linux (x86_64)](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip)，这里的是5.1.0版本的

然后，解压

> **注意：路径内不能存在中文，否则 Isaac Sim 会闪退。**  

## Ros 2 与 Isaac Sim连接

运行 `isaac-sim.selector.sh`

ROS Bridge Extension 选择 `isaacsim.ros2.bridge`

Use Internal Ros2 Libraries 选择 `humble`

点击 `Start`

Isaac Sim第一次启动会有些慢

然后，Create -> Graphs -> Action Graph

在这个 `Action Graph` 绘出下图

![Action Graph](/imgs/image1.png)

然后，点击 `play`

![点击 `play`](/imgs/image2.png)

在 Linux 命令行之内运行`ros2 topic echo /clock`，最后会出现时间信息，说明连接成功

![Linux 命令行之内运行`ros2 topic echo /clock`](/imgs/image3.png)

***

## 参考资料

<span id="footnote-1" style="font-size: var(--global-font-size);">1</span>. [ubuntu22.04物理机双系统手动分区](https://blog.csdn.net/itas109/article/details/136995139)