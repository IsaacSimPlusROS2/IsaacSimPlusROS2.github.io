---
title: Berkeley Humanoid Lite Lowlevel 的 robot/imu.py
date: 2026-02-10 20:28:09
tags: [Berkeley Humanoid Lite, Python]
categories: [Berkeley Humanoid Lite]
---

## **核心逻辑**

*   **通信协议**：IMU 通过串口发送二进制数据帧。每一帧以 `0x55` 开头，紧接着是**帧类型**（Frame Type），然后是数据载荷。
*   **数据解析 (`__read_frame`)**：
    *   代码不断读取串口数据，利用 `struct.unpack` 将二进制数据转换为整数。
    *   **物理量转换**：原始整数数据被转换为物理单位。例如，加速度的原始值除以 `32768.0` 再乘以量程 `16.0`，得到单位为 $g$ 的加速度。
    *   支持的数据类型包括：加速度、角速度、欧拉角（Roll/Pitch/Yaw）、磁场、四元数。
*   **多线程运行 (`run_forever`)**：
    *   为了不阻塞主程序的运行，驱动开启了一个后台线程（`threading.Thread`）专门负责读取串口数据并更新类的成员变量（如 `self.acceleration`）。
*   **配置机制**：
    *   IMU 有写保护。修改配置前必须先调用 `unlock()` 发送特定密钥。
    *   修改后需调用 `save()` 将配置写入 EEPROM，否则掉电丢失。

### **主要类 `SerialImu` 的关键方法**

*   **`__init__`**: 打开串口，初始化存储数据的 numpy 数组。
*   **`__read_frame`**: 解析协议的核心。识别帧头 `0x55`，根据帧类型更新对应的属性（如 `self.angle`）。
*   **`unlock`**: 写入寄存器 `0x69` 值为 `0xB588`，这是该硬件解除写保护的标准操作。
*   **`set_output_content`**: 配置 IMU 每一轮发送哪些数据包（例如只发加速度和角速度，不发 GPS），通过位掩码实现。
*   **`set_baudrate` / `set_sampling_rate`**: 修改通信速率和数据更新频率。

## Class 常量列表

### Class `ImuRegisters` (寄存器地址)

该类定义了 IMU 内部寄存器的地址映射，用于配置和读取状态。

| 分类 | 寄存器名称 | 地址 (Hex) | 描述 |
| :--- | :--- | :--- | :--- |
| **系统控制** | `SAVE` | `0x00` | 保存配置/重启/恢复出厂 |
| | `CALSW` | `0x01` | 进入校准模式 |
| | `RSW` | `0x02` | 配置输出内容 (位掩码) |
| | `RRATE` | `0x03` | 输出速率 (采样率) |
| | `BAUD` | `0x04` | 串口波特率 |
| | `SLEEP` | `0x22` | 休眠模式 |
| | `ORIENT` | `0x23` | 安装方向设置 |
| | `POWONSEND`| `0x2D` | 上电是否自动发送 |
| | `KEY` | `0x69` | **解锁密钥** (写入 0xB588 解锁) |
| **零偏校准** | `AXOFFSET` - `AZOFFSET` | `0x05` - `0x07` | 加速度 X/Y/Z 零偏 |
| | `GXOFFSET` - `GZOFFSET` | `0x08` - `0x0A` | 角速度 X/Y/Z 零偏 |
| | `HXOFFSET` - `HZOFFSET` | `0x0B` - `0x0D` | 磁场 X/Y/Z 零偏 |
| **传感器配置** | `MAGRANGX` - `MAGRANGZ`| `0x1C` - `0x1E` | 磁场校准范围 |
| | `BANDWIDTH` | `0x1F` | 带宽设置 |
| | `GYRORANGE` | `0x20` | 陀螺仪量程 |
| | `ACCRANGE` | `0x21` | 加速度计量程 |
| | `AXIS6` | `0x24` | 算法选择 (6轴/9轴) |
| | `FILTK` | `0x25` | 动态滤波系数 |
| | `ACCFILT` | `0x2A` | 加速度滤波器 |
| **数据读取** | `AX`, `AY`, `AZ` | `0x34` - `0x36` | 加速度原始数据 |
| | `GX`, `GY`, `GZ` | `0x37` - `0x39` | 角速度原始数据 |
| | `HX`, `HY`, `HZ` | `0x3A` - `0x3C` | 磁场原始数据 |
| | `ROLL`, `PITCH`, `YAW` | `0x3D` - `0x3F` | 欧拉角 |
| | `TEMP` | `0x40` | 温度 |
| | `Q0` - `Q3` | `0x51` - `0x54` | 四元数 |
| **GPS/高度** | `PRESSUREL/H` | `0x45`/`0x46` | 气压值 |
| | `HEIGHTL/H` | `0x47`/`0x48` | 高度值 |
| | `LONL/H`, `LATL/H` | `0x49`-`0x4C` | 经纬度 |
| | `GPSHEIGHT`, `GPSYAW` | `0x4D`, `0x4E` | GPS海拔与航向 |
| | `GPSVL/H` | `0x4F`, `0x50` | GPS地速 |
| **I/O 与其他** | `D0MODE` - `D3MODE` | `0x0E` - `0x11` | 数字引脚模式 |
| | `IICADDRESS` | `0x1A` | I2C 地址 |
| | `LEDOFF` | `0x1B` | 关闭 LED |
| | `READADDR` | `0x27` | 读寄存器指令 |
| | `VERSION` | `0x2E` | 固件版本 |

---

### Class `FrameType` (返回数据帧类型)

IMU 发回的数据包中，标识该包属于哪种数据的标识符（位于包头的 `0x55` 之后）。

| 常量名称 | 值 (Hex) | 对应数据内容 |
| :--- | :--- | :--- |
| `TIME` | `0x50` | 时间戳 (年/月/日/时/分/秒/毫秒) |
| `ACCELERATION` | `0x51` | 加速度 (X, Y, Z) & 温度 |
| `ANGULAR_VELOCITY` | `0x52` | 角速度 (X, Y, Z) |
| `ANGLE` | `0x53` | 欧拉角 (Roll, Pitch, Yaw) |
| `MAGNETIC_FIELD` | `0x54` | 磁场 (X, Y, Z) |
| `PORT_STATUS` | `0x55` | 端口状态 (数字IO电平) |
| `BAROMETER_ALTITUDE` | `0x56` | 气压与高度 |
| `LATITUDE_LONGITUDE` | `0x57` | GPS 经纬度 |
| `GROUND_SPEED` | `0x58` | GPS 地速 |
| `QUATERNION` | `0x59` | 四元数 (Q0, Q1, Q2, Q3) |
| `GPS_POSITION_ACCURACY`| `0x5A` | GPS 定位精度 |
| `READY` | `0x5F` | 传感器就绪/响应 |

---

### Class `SamplingRate` (采样率设置)

用于写入 `RRATE` 寄存器，控制数据回传频率。

| 常量名称 | 值 (Hex) | 频率 |
| :--- | :--- | :--- |
| `RATE_0_2_HZ` | `0x01` | 0.2 Hz |
| `RATE_0_5_HZ` | `0x02` | 0.5 Hz |
| `RATE_1_HZ` | `0x03` | 1 Hz |
| `RATE_2_HZ` | `0x04` | 2 Hz |
| `RATE_5_HZ` | `0x05` | 5 Hz |
| `RATE_10_HZ` | `0x06` | 10 Hz |
| `RATE_20_HZ` | `0x07` | 20 Hz |
| `RATE_50_HZ` | `0x08` | 50 Hz |
| `RATE_100_HZ` | `0x09` | 100 Hz |
| `RATE_200_HZ` | `0x0B` | 200 Hz |
| `RATE_SINGLE` | `0x0C` | 单次请求 |
| `RATE_NO_RETURN`| `0x0D` | 不自动回传 |

---

### Class `Baudrate` (波特率设置)

用于写入 `BAUD` 寄存器以及初始化 `serial.Serial`。

| 常量名称 | 值 (Hex) | 实际波特率 (bps) |
| :--- | :--- | :--- |
| `BAUD_4800` | `0x01` | 4800 |
| `BAUD_9600` | `0x02` | 9600 |
| `BAUD_19200` | `0x03` | 19200 |
| `BAUD_38400` | `0x04` | 38400 |
| `BAUD_57600` | `0x05` | 57600 |
| `BAUD_115200` | `0x06` | 115200 (默认常见) |
| `BAUD_230400` | `0x07` | 230400 |
| `BAUD_460800` | `0x08` | 460800 |

## 使用示例

在代码底部的 `if __name__ == "__main__":` 块展示了标准用法：

1.  **初始化**：`imu = SerialImu(baudrate=Baudrate.BAUD_460800)`
2.  **配置（可选）**：
    *   先 `unlock()`。
    *   设置参数（如 `set_output_content` 开启四元数输出）。
    *   `save()` 保存配置。
3.  **运行**：
    *   `imu.run_forever()`：启动后台线程读取数据。
    *   主循环中使用 `imu.acceleration`, `imu.quaternion` 等属性获取最新的传感器数据。