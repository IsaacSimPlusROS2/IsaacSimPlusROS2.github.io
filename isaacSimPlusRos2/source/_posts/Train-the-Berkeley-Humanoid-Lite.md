---
title: 在 Isaac Sim 里面训练 Berkeley Humanoid Lite
date: 2026-02-10 11:23:52
tags: [Berkeley Humanoid Lite, Python]
categories: Berkeley Humanoid Lite
---

## 前期提要

在Berkeley Humanoid Lite的[getting-started-with-software/training-environment](https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-software/training-environment)中，给出了安装 Training Environment 的办法。

[getting-started-with-software](https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-software/training-environment)部分主要讲述了一件事：**通过不断的“试错”，把一个只会原地瘫倒的机器人 3D 模型，训练成一个能跑、能跳、能平衡的“智能体”**。

这里就不说了，我写一些我遇到的问题的解决方法

## Carb 库丢失

遇到的错误 `No module named 'carb._carb'` 和 `TypeError: 'NoneType' object is not callable`

```
(berkeley-humanoid-lite) mryan2005@venti:~/berkeley_humanoid_isaac/Berkeley-Humanoid-Lite$ python ./scripts/rsl_rl/play.py --task Velocity-Berkeley-Humanoid-Lite-v0 --num_envs 16
[Warning] Unable to expose 'isaacsim.simulation_app' API: No module named 'carb._carb'
[INFO][AppLauncher]: Using device: cuda:0
[INFO][AppLauncher]: Loading experience file: /home/mryan2005/IsaacLab/apps/isaacsim_4_5/isaaclab.python.kit
Traceback (most recent call last):
File "/home/mryan2005/berkeley_humanoid_isaac/Berkeley-Humanoid-Lite/./scripts/rsl_rl/play.py", line 31, in <module>
app_launcher = AppLauncher(args_cli)
File "/home/mryan2005/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py", line 131, in init
self._create_app()
File "/home/mryan2005/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py", line 823, in _create_app
self._app = SimulationApp(self._sim_app_config, experience=self._sim_experience_file)
TypeError: 'NoneType' object is not callable
```

### 核心原因

Python 环境没有正确加载 Isaac Sim 的底层库。

### 修复环境链接

我的 Isaac Sim 5.1 的安装路径是在 `/home/mryan2005/isaac-sim-standalone-5.1.0-linux-x86_64`

### 重新配置 Isaac Lab 链接

在 IsaacLab 根目录下执行

```bash
cd ~/IsaacLab
# 将路径替换为你实际的 Isaac Sim 5.1 路径
ln -sfn /home/mryan2005/isaac-sim-standalone-5.1.0-linux-x86_64 _isaac_sim
```

## flatdict 无法安装

```bash
# 1. 下载 flatdict 源码包
wget https://pypi.tuna.tsinghua.edu.cn/packages/3e/0d/424de6e5612f1399ff69bf86500d6a62ff0a4843979701ae97f120c7f1fe/flatdict-4.0.1.tar.gz

# 2. 解压
tar -xvf flatdict-4.0.1.tar.gz
cd flatdict-4.0.1

# 3. [核心] 使用对应的 Python 解释器直接运行安装脚本
# 如果是装给 Isaac Sim 内部：
/home/mryan2005/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh setup.py install

# 如果是装给 Conda 环境：
# python setup.py install

# 4. 检查是否成功
cd ..
rm -rf flatdict-4.0.1*
```

## numpy、tensorboard 不存在

```bash
# 1. 确保已彻底退出 Conda (直到命令提示符前没有 (base) 等字样)
conda deactivate

# 2. 定义 Isaac Sim 路径
export ISAACSIM_PATH="/home/mryan2005/isaac-sim-standalone-5.1.0-linux-x86_64"

# 4. 使用禁用隔离模式安装 flatdict 到内部环境
$ISAACSIM_PATH/python.sh -m pip install numpy tensorboard --no-build-isolation

# 5. 验证
$ISAACSIM_PATH/python.sh -c "import numpy; print('Isaac Sim 内部 numpy 安装成功')"
```

## 路径污染

### 编写 run_play.sh

```bash
#!/bin/bash

export ISAACSIM_PATH="/home/mryan2005/isaac-sim-standalone-5.1.0-linux-x86_64"
export ISAAC_LAB_PATH="/home/mryan2005/IsaacLab"
export PROJECT_ROOT="/home/mryan2005/berkeley_humanoid_isaac/Berkeley-Humanoid-Lite"

unset PYTHONPATH
unset CONDA_PREFIX
unset PYTHONHOME

# 1. 删除 Omniverse 通用缓存 (包含旧版扩展记录)
rm -rf ~/.cache/ov
rm -rf ~/.cache/nvidia/GLCache

# 2. 删除着色器缓存 (强制 5.1 重新生成)
rm -rf ~/.nvidia-shader-cache

# 3. 删除 Isaac Sim 5.1 的运行缓存（需要修改路径！！！！）
rm -rf ~/isaac-sim-standalone-5.1.0-linux-x86_64/kit/data/Kit/Isaac-Sim/5.1/cache

export PYTHONPATH=$PROJECT_ROOT/source/berkeley_humanoid_lite
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/source/berkeley_humanoid_lite_assets
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/isaaclab
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/extensions/omni.isaac.lab_rl
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/extensions/omni.isaac.lab_tasks

export ISAACLAB_EXP_FILE="$ISAACSIM_PATH/apps/isaacsim.python.kit"

cd $PROJECT_ROOT

$ISAACSIM_PATH/python.sh ./scripts/rsl_rl/play.py \
    --task Velocity-Berkeley-Humanoid-Lite-v0 \
    --num_envs 1
```

### ## 编写 run_train.sh

```bash
#!/bin/bash

export ISAACSIM_PATH="/home/mryan2005/isaac-sim-standalone-5.1.0-linux-x86_64"
export ISAAC_LAB_PATH="/home/mryan2005/IsaacLab"
export PROJECT_ROOT="/home/mryan2005/berkeley_humanoid_isaac/Berkeley-Humanoid-Lite"

export PYTHONPATH=""
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/source/berkeley_humanoid_lite
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/source/berkeley_humanoid_lite_assets
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/source/berkeley_humanoid_lite_lowlevel

export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/isaaclab
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/extensions/omni.isaac.lab_rl
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/extensions/omni.isaac.lab_tasks

export ISAACLAB_EXP_FILE="$ISAAC_LAB_PATH/apps/isaaclab.python.kit"

# 1. 删除 Omniverse 通用缓存 (包含旧版扩展记录)
rm -rf ~/.cache/ov
rm -rf ~/.cache/nvidia/GLCache

# 2. 删除着色器缓存 (强制 5.1 重新生成)
rm -rf ~/.nvidia-shader-cache

# 3. 删除 Isaac Sim 5.1 的运行缓存（需要修改路径！！！！）
rm -rf ~/isaac-sim-standalone-5.1.0-linux-x86_64/kit/data/Kit/Isaac-Sim/5.1/cache

cd $PROJECT_ROOT

$ISAACSIM_PATH/python.sh ./scripts/rsl_rl/train.py \
    --task Velocity-Berkeley-Humanoid-Lite-v0 \
    --num_envs 1024 \
    "$@"      # 确保后面有 "$@"，这样在外面输入的 --resume 才能传给 python
```

## 版本残留

Isaac Lab 在安装时会扫描 _isaac_sim 链接并生成固定的路径映射。

运行下面的重置 Isaac Lab 的版本映射的命令即可

```bash
# 1. 进入 IsaacLab 目录
cd ~/IsaacLab

# 2. 强制更正软链接 (确保指向 5.1)
ln -sfn /home/mryan2005/isaac-sim-standalone-5.1.0-linux-x86_64 _isaac_sim

# 3. 彻底删除旧版配置残留 (如果 5.1 的配置文件夹没生成，就从 4.5 复制一份)
if [ ! -d "apps/isaacsim_5_1" ]; then
    cp -r apps/isaacsim_4_5 apps/isaacsim_5_1
fi

# 4. [关键] 强制重新安装 Isaac Lab 核心
# 使用 --force-reinstall 确保 egg-info 里的路径元数据被重写
conda activate berkeley-humanoid-lite
pip install -e source/isaaclab --force-reinstall --no-build-isolation
```

## RSL-RL 算法库的结构性矛盾

不使用 Isaac Lab 自动安装的 RSL-RL 4.0就可以了

在 Isaac Sim 内部环境中强制降级，以匹配机器人的 PPO 配置。

```bash
# 确保已 conda deactivate，在终端 A 执行
export ISAACSIM_PATH="/home/mryan2005/isaac-sim-standalone-5.1.0-linux-x86_64"

# 1. 卸载新版
$ISAACSIM_PATH/python.sh -m pip uninstall -y rsl-rl-lib
# 2. 安装完全匹配的旧版
$ISAACSIM_PATH/python.sh -m pip install git+https://github.com/leggedrobotics/rsl_rl.git@v1.0.2
```

## Isaac Lab 0.54+ API 的剧烈变动

### 修改 ./scripts/rsl_rl/play.py 的代码

其实，./scripts/rsl_rl/play.py的代码在 Isaac Sim 5.1 下是有问题的，所以，得修改成这样

```python
# scripts/rsl_rl/play.py

import argparse
import os
import sys

# --- [关键] 1. 初始化 AppLauncher (必须最先导入) ---
from isaaclab.app import AppLauncher

# 将当前目录加入路径以便导入 cli_args.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cli_args

# 解析参数
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="是否录制视频")
parser.add_argument("--video_length", type=int, default=200, help="录制视频步数")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="禁用 Fabric")
parser.add_argument("--num_envs", type=int, default=None, help="仿真环境数量")
parser.add_argument("--task", type=str, default=None, help="任务名称")

# 添加 RSL-RL 和 AppLauncher 的参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动仿真 App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- [关键] 2. 仿真启动后的导入 ---
import gymnasium as gym
import torch
from omegaconf import OmegaConf

from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.string as string_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# 导入自定义任务包以注册环境
import berkeley_humanoid_lite.tasks  # noqa: F401

def main():
    """推理主函数"""
    
    # 1. 解析环境配置
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else 16, 
        use_fabric=not args_cli.disable_fabric
    )

    # 2. 解析 RSL-RL Agent 配置
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 3. 寻找模型路径
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    resume_path = None
    try:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] 找到模型文件: {resume_path}")
    except Exception as e:
        print(f"[WARN] 未能自动找到模型: {e}")
        fallback_path = os.path.join("checkpoints", "policy_humanoid.pt")
        if os.path.exists(fallback_path):
            resume_path = fallback_path
            print(f"[INFO] 使用备用模型: {resume_path}")
        else:
            print("[INFO] 警告：未找到模型权重，进入“零动作测试模式”。")

    # 4. 创建环境
    render_mode = "rgb_array" if args_cli.video else None
    print(f"[INFO] 正在创建环境: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # 5. 录制视频包装
    if args_cli.video and resume_path:
        log_dir = os.path.dirname(resume_path)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 6. 环境包装
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # RslRlVecEnvWrapper 会规范化输入输出，供 RSL-RL 使用
    env = RslRlVecEnvWrapper(env)

    # --- 获取精确的维度信息 ---
    num_envs = env.num_envs
    num_actions = env.num_actions
    device = env.unwrapped.device

    # 7. 策略实例化
    if resume_path:
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        policy = ppo_runner.get_inference_policy(device=device)
    else:
        print(f"[INFO] 运行零动作策略 (维度: {num_envs} envs x {num_actions} actions)...")
        # 修复点：显式使用正确的维度 (num_envs, num_actions)
        def policy(obs):
            return torch.zeros((num_envs, num_actions), device=device)

    # 8. 仿真循环
    print(f"[INFO] 启动成功，正在运行...")
    
    # 获取初始观测 (RslRlVecEnvWrapper 返回单个 Tensor)
    obs = env.get_observations()
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # 策略推理
            actions = policy(obs)
            # 步进：返回 obs, rewards, terminations, extras
            obs, rewards, terminations, extras = env.step(actions)

    # 9. 资源清理
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
```

### 修改 ./scripts/rsl_rl/train.py 的代码

```python
# scripts/rsl_rl/train.py

import argparse
import os
import sys

# --- [关键] 1. 初始化 AppLauncher (必须最先导入) ---
from isaaclab.app import AppLauncher

# 将当前目录加入路径以便导入 cli_args.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cli_args

# 解析参数
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="是否在训练中录制视频")
parser.add_argument("--video_length", type=int, default=200, help="录制视频步数")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="禁用 Fabric")
parser.add_argument("--num_envs", type=int, default=None, help="仿真环境数量")
parser.add_argument("--task", type=str, default=None, help="任务名称")
parser.add_argument("--seed", type=int, default=None, help="随机种子")

# 添加 RSL-RL 和 AppLauncher 的参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动仿真 App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- [关键] 2. 仿真启动后的导入 ---
import gymnasium as gym
import datetime
import pickle
import torch
from omegaconf import OmegaConf

from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.dict as dict_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

# 导入自定义任务包以注册环境
import berkeley_humanoid_lite.tasks  # noqa: F401

def main():
    """训练主函数"""
    
    # 1. 解析环境配置
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else 4096, 
        use_fabric=not args_cli.disable_fabric
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # 2. 解析 RSL-RL Agent 配置
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 3. 设置日志目录
    log_dir = os.path.join(
        "logs", "rsl_rl", agent_cfg.experiment_name, 
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)

    # 4. 创建环境
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # 5. 录制视频包装
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % 1000 == 0, # 每 1000 步录一段
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 6. 环境包装
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # 7. 保存配置 (修复之前报错的关键点)
    # 使用 OmegaConf 和 Pickle 替代已删除的 dump_yaml/dump_pickle
    config_dir = os.path.join(log_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    # 导出 Agent 配置
    with open(os.path.join(config_dir, "agent.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(agent_cfg))
    
    # 导出环境配置 (Pickle 备份)
    with open(os.path.join(config_dir, "env_cfg.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)

    # 8. 初始化并运行训练
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 如果指定了 resume，则加载旧模型
    if agent_cfg.resume:
        from isaaclab_tasks.utils import get_checkpoint_path
        resume_path = get_checkpoint_path(
            os.path.join("logs", "rsl_rl", agent_cfg.experiment_name), 
            agent_cfg.load_run, 
            agent_cfg.load_checkpoint
        )
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        runner.load(resume_path)

    print(f"[INFO] 训练正式开始...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 9. 资源清理
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
```

## 硬件与底层渲染障碍

其实，这个部分要解决也很简单的，就是就该参数`num_envs`就可以了，一般情况下，按照这样的阶梯设置`512`、`1024`、`2048`、`4096`来设置。