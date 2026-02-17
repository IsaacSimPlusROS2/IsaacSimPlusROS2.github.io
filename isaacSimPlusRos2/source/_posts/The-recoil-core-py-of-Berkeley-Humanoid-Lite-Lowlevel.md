---
title: Berkeley Humanoid Lite Lowlevel 的 recoil/core.py
date: 2026-02-10 20:37:37
tags: [Berkeley Humanoid Lite, Python]
categories: [Berkeley Humanoid Lite]
---

根据您提供的代码，我将其中的常量类整理为表格，并对 `Bus` 类的核心方法进行了分类列表和说明。

## 常量类列表

### Class `Function` (CAN功能码)

该类定义了CAN ID的高4位，用于区分消息类型。

| 常量名称 | 值 (Binary) | 值 (Decimal) | 描述 |
| :--- | :--- | :--- | :--- |
| `NMT` | `0b0000` | 0 | 网络管理 (模式切换) |
| `SYNC_EMCY` | `0b0001` | 1 | 同步/紧急消息 |
| `TIME` | `0b0010` | 2 | 时间戳 |
| `TRANSMIT_PDO_1` | `0b0011` | 3 | 发送 PDO 1 |
| `RECEIVE_PDO_1` | `0b0100` | 4 | 接收 PDO 1 |
| `TRANSMIT_PDO_2` | `0b0101` | 5 | 发送 PDO 2 (位置/速度反馈) |
| `RECEIVE_PDO_2` | `0b0110` | 6 | 接收 PDO 2 (位置/速度命令) |
| `TRANSMIT_PDO_3` | `0b0111` | 7 | 发送 PDO 3 |
| `RECEIVE_PDO_3` | `0b1000` | 8 | 接收 PDO 3 |
| `TRANSMIT_PDO_4` | `0b1001` | 9 | 发送 PDO 4 |
| `RECEIVE_PDO_4` | `0b1010` | 10 | 接收 PDO 4 |
| `TRANSMIT_SDO` | `0b1011` | 11 | 发送 SDO (读写参数响应) |
| `RECEIVE_SDO` | `0b1100` | 12 | 接收 SDO (读写参数请求) |
| `FLASH` | `0b1101` | 13 | Flash 操作 |
| `HEARTBEAT` | `0b1110` | 14 | 心跳包 |

---

### Class `Mode` (控制模式)

定义电机的工作状态。

| 分类 | 常量名称 | 值 (Hex) | 描述 |
| :--- | :--- | :--- | :--- |
| **安全模式** | `DISABLED` | `0x00` | 禁用/停机 |
| | `IDLE` | `0x01` | 空闲 |
| **特殊模式** | `DAMPING` | `0x02` | 阻尼模式 |
| | `CALIBRATION` | `0x05` | 校准模式 |
| **闭环模式** | `CURRENT` | `0x10` | 电流(力矩)闭环 |
| | `TORQUE` | `0x11` | 力矩闭环 |
| | `VELOCITY` | `0x12` | 速度闭环 |
| | `POSITION` | `0x13` | 位置闭环 |
| **开环模式** | `VABC_OVERRIDE` | `0x20` | VABC 电压直接控制 |
| | `VALPHABETA_OVERRIDE` | `0x21` | Vαβ 电压直接控制 |
| | `VQD_OVERRIDE` | `0x22` | Vdq 电压直接控制 |
| **调试** | `DEBUG` | `0x80` | 调试模式 |

---

### Class `ErrorCode` (错误代码)

位掩码（Bitmask），支持多种错误同时存在。

| 常量名称 | 值 (Binary) | 描述 |
| :--- | :--- | :--- |
| `NO_ERROR` | `0...0000` | 无错误 |
| `GENERAL` | `...0001` | 通用错误 |
| `ESTOP` | `...0010` | 急停 |
| `INITIALIZATION_ERROR` | `...0100` | 初始化错误 |
| `CALIBRATION_ERROR` | `...1000` | 校准错误 |
| `POWERSTAGE_ERROR` | `..10000` | 功率级错误 |
| `INVALID_MODE` | `..100000` | 无效模式 |
| `WATCHDOG_TIMEOUT` | `..1000000` | 看门狗超时 |
| `OVER_VOLTAGE` | `..10000000` | 过压 |
| `OVER_CURRENT` | `..100000000` | 过流 |
| `OVER_TEMPERATURE` | `..1000000000` | 过温 |
| `CAN_TX_FAULT` | `..10000000000` | CAN发送故障 |
| `I2C_FAULT` | `..100000000000` | I2C通信故障 |

---

### Class `Parameter` (参数地址表)

由于参数众多，按功能模块分组列出主要范围。

| 功能模块 | 地址范围 (Hex) | 包含的主要参数 |
| :--- | :--- | :--- |
| **系统信息** | `0x000` - `0x014` | 设备ID、固件版本、看门狗、CAN频率、当前模式、错误码 |
| **位置控制器** | `0x018` - `0x070` | 减速比、PID参数(Kp/Ki)、限制(力矩/速度/位置)、目标值/测量值、滤波器系数 |
| **电流控制器(FOC)** | `0x074` - `0x0D4` | 电流限制、PID参数、各相电流/电压(ABC/dq轴)的测量值与设定值、积分器 |
| **功率级(PowerStage)** | `0x0D8` - `0x100` | 定时器值、ADC原始读数/偏移、欠压/过压阈值、母线电压 |
| **电机参数** | `0x104` - `0x110` | 极对数、力矩常数、相序、最大校准电流 |
| **编码器(Encoder)** | `0x114` - `0x140` | I2C缓冲/计数、CPR(分辨率)、位置偏移、速度滤波、位置/速度读数、磁通偏移 |

---

## Class `Bus` 方法列表

`Bus` 类负责底层的 CAN 通信以及协议的封装。

### 基础通信与管理

| 方法名 | 参数 | 描述 |
| :--- | :--- | :--- |
| `__init__` | `channel`, `bitrate` | 初始化 CAN 总线 (默认 socketcan, 1Mbps)。 |
| `stop` | - | 关闭 CAN 总线连接。 |
| `receive` | `filter_device_id`, `filter_function`, `timeout` | 接收并解析一帧 CAN 数据，支持按设备ID或功能ID过滤。 |
| `transmit` | `frame` | 将 `CANFrame` 对象打包并发送到总线。 |
| `ping` | `device_id`, `timeout` | 发送测试包并等待特定响应，检测设备是否在线。 |
| `feed` | `device_id` | 发送心跳包 (`HEARTBEAT`)，防止设备看门狗超时。 |
| `set_mode` | `device_id`, `mode` | 发送 NMT 命令切换设备运行模式 (参考 `Mode` 类)。 |

### Flash 存储操作

| 方法名 | 参数 | 描述 |
| :--- | :--- | :--- |
| `load_settings_from_flash` | `device_id` | 命令设备从 Flash 读取保存的配置。 |
| `store_settings_to_flash` | `device_id` | 命令设备将当前配置保存到 Flash。 |

### 底层参数读写 (SDO)

这些是实现参数存取的私有或底层方法，通常不直接调用，而是通过具体的属性方法调用。

| 方法名 | 描述 |
| :--- | :--- |
| `_read_parameter` | 发送读请求，返回包含数据的帧。 |
| `_write_parameter` | 发送写请求。 |
| `_read_parameter_[type]` | 包含 `_bytes`, `_f32`, `_i32`, `_u32`。读取并自动转换数据类型。 |
| `_write_parameter_[type]` | 包含 `_bytes`, `_f32`, `_i32`, `_u32`。将数据转换类型后写入。 |

### 自动化配置助手 (Helper Functions)

根据物理公式自动计算并设置控制参数，简化调参过程。

| 方法名 | 作用 |
| :--- | :--- |
| `set_current_bandwidth` | 根据带宽(Hz)、电阻、电感计算并设置电流环 **Kp, Ki**。 |
| `set_torque_bandwidth` | 根据带宽计算并设置力矩滤波系数 **Alpha**。 |
| `set_bus_voltage_bandwidth`| 根据带宽计算并设置母线电压滤波系数 **Alpha**。 |
| `set_encoder_velocity_bandwidth` | 根据带宽计算并设置编码器速度滤波系数 **Alpha**。 |

### 实时控制 (PDO)

用于高频闭环控制的方法。

| 方法名 | 参数 | 描述 |
| :--- | :--- | :--- |
| `transmit_pdo_2` | `position_target`, `velocity_target` | 发送位置和速度目标值 (发送控制指令)。 |
| `receive_pdo_2` | - | 接收当前测量的位置和速度 (获取反馈)。 |
| `write_read_pdo_2` | `position_target`, `velocity_target` | **原子操作**：先发送控制指令，紧接着等待并返回反馈数据。常用于控制循环中。 |

### 具体参数读写接口 (Properties)

`Bus` 类中包含了大量针对 `Parameter` 类中定义的具体参数的读写方法。为节省篇幅，此处按命名规则总结：

*   **命名规则**: `read_[参数名](device_id)` 和 `write_[参数名](device_id, value)`
*   **示例**:
    *   `read_position_kp` / `write_position_kp` (读写位置环 P 增益)
    *   `read_current_limit` / `write_current_limit` (读写电流限制)
    *   `read_motor_pole_pairs` / `write_motor_pole_pairs` (读写极对数)
*   **覆盖范围**: 几乎涵盖了 `Parameter` 表中定义的所有可配置项。