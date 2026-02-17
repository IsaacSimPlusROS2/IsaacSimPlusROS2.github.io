---
title: Berkeley Humanoid Lite Lowlevel 的 recoil/can.py
date: 2026-02-10 20:38:00
tags: [Berkeley Humanoid Lite, Python]
categories: [Berkeley Humanoid Lite]
---

## `DataFrame` 的定义

一个通用的数据帧结构，用于存储通信协议中的基本信息，并强制执行数据长度的一致性检查。

```python
class DataFrame:
    def __init__(
        self,
        device_id: int = 0,
        func_id: int | None = None,
        size: int = 0,
        data: bytes | bytearray = b""
    ):
        self.device_id = device_id
        self.func_id = func_id
        self.size = size
        self.data = data
        assert self.size == len(self.data)
```

参数详解：

- `device_id`: 设备 ID（整数），用于标识发送者或接收者。
- `func_id`: 功能 ID（整数或 None）。用于标识这条指令是做什么的（例如：读数据、写数据、心跳包）。
- `size`：数据长度（整数）。声明 `data` 中包含多少字节。
- `data`：实际的数据载荷，类型可以是 `bytes` 或 `bytearray`，表明它可以是不可变的字节串（`b'\x01'`）或可变的字节数组。

`assert self.size == len(self.data)` 这一行代码确保了 `size` 参数与 `data` 的实际长度一致，防止数据不完整或过长的情况发生。

## `CANFrame` 的定义

### 三个常量定义

```python
class CANFrame(DataFrame):
    ID_STANDARD = 0
    ID_EXTENDED = 1

    DEVICE_ID_MSK = 0x7F
    FUNC_ID_POS = 7
    FUNC_ID_MSK = 0x0F << FUNC_ID_POS
```

`ID_STANDARD` / `ID_EXTENDED`：定义 CAN 帧 ID 的类型。
- 0 (标准帧)：使用 11 位标识符（CAN 2.0A）。
- 1 (扩展帧)：使用 29 位标识符（CAN 2.0B）。

`DEVICE_ID_MSK`：设备 ID 的掩码，`0x7F` 表示设备 ID 只能占用 `7` 位（`0`-`127`），设备ID的范围是 `0` 到 `127`。
`FUNC_ID_POS`：功能 ID 在 CAN 帧中的位置，定义为 `7`，表示功能 ID 从第 `7` 位开始（即第 8 个比特，因为从 0 开始计数）。
`FUNC_ID_MSK`：功能 ID 的掩码，`0x0F` 的二进制是 `1111`（`4`个比特），`<< 7` 表示左移 `7` 位，结果掩码覆盖了 ID 的 高 `4` 位（第 `7` 到 `10` 位），这意味着 功能ID 占据了 `4` 个比特。

### `__init__` 方法

```python
def __init__(
        self,
        device_id: int = 0,
        func_id: int | None = None,
        size: int = 0,
        data: bytes = b""
    ):
        super().__init__(device_id, func_id, size, data)
        assert self.size <= 8
```

参数列表：

- `device_id`：设备标识符，默认为 0。
- `func_id`：功能标识符，类型提示允许是 `int` 或 `None`。
- `size`：数据长度，默认为 0。
- `data`：实际的数据载荷，默认为空字节串 (b"")。

`super().__init__(device_id, func_id, size, data)` 调用父类 `DataFrame` 的构造函数，初始化基本属性。

`assert self.size <= 8` 确保 CAN 帧的数据长度不超过 8 字节，这是 CAN 协议的限制。