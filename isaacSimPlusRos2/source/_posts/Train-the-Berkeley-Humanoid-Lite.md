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

$ISAACSIM_PATH/python.sh ./scripts/rsl_rl/playNew.py \
    --task Velocity-Berkeley-Humanoid-Lite-v0 \
    --num_envs 1 \
    --load_run 2026-02-10_15-54-39
```

### 编写 run_train.sh

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

### 修改 ./scripts/rsl_rl/playNew.py 的代码

我们可以按照 train.py 来写一个 playNew.py 的代码，从而可以看到训练结果

```python
"""
使用 RSL-RL v1.0.2 推理/预览 Berkeley Humanoid Lite 的脚本。
结构严格仿照 train.py，适配 Isaac Sim 5.1。
"""

import argparse
import os
import sys
import torch

# --- [1] 初始化 AppLauncher (必须最先导入) ---
from isaaclab.app import AppLauncher

# 解决本地导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play/Inference with RSL-RL agent.")
parser.add_argument("--num_envs", type=int, default=1, help="仿真环境数量(预览通常为1)")
parser.add_argument("--task", type=str, default=None, help="任务名称")
parser.add_argument("--seed", type=int, default=None, help="随机种子")
# 添加 RSL-RL 参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 强制开启图形界面
args_cli.headless = False

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- [2] 仿真启动后的导入 ---
import gymnasium as gym
from omegaconf import OmegaConf

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.dict import class_to_dict

import berkeley_humanoid_lite.tasks  # noqa: F401

# --- [兼容层] 复用 Wrapper ---
class RslRlVecEnvWrapperFixed(RslRlVecEnvWrapper):
    def __init__(self, env):
        self.env = env
        base_env = env.unwrapped
        self.unwrapped_env = base_env 
        self.device = base_env.device
        
        obs_manager = base_env.observation_manager
        
        def _to_int(val):
            if isinstance(val, (tuple, list)): return int(val[0])
            return int(val)

        self.num_obs = _to_int(obs_manager.group_obs_dim["policy"])
        self.num_actions = _to_int(base_env.action_manager.total_action_dim)
        self.num_envs = base_env.num_envs

        if "critic" in obs_manager.group_obs_dim:
            self.num_privileged_obs = _to_int(obs_manager.group_obs_dim["critic"])
        else:
            self.num_privileged_obs = None
            
        self.episode_length_buf = base_env.episode_length_buf

    def _strip_tensordict(self, obs):
        if obs is None: return None
        if not isinstance(obs, torch.Tensor):
            if isinstance(obs, dict):
                return torch.cat(list(obs.values()), dim=-1)
        return obs.view(obs.shape)

    def get_observations(self):
        obs = self.unwrapped.observation_manager.compute_group("policy")
        return self._strip_tensordict(obs)

    def get_privileged_observations(self):
        if self.num_privileged_obs is not None:
            obs = self.unwrapped.observation_manager.compute_group("critic")
            return self._strip_tensordict(obs)
        return None

    def step(self, actions):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs = self._strip_tensordict(obs_dict["policy"])
        privileged_obs = None
        if "critic" in obs_dict:
            privileged_obs = self._strip_tensordict(obs_dict["critic"])
        elif self.num_privileged_obs is not None:
            privileged_obs = self.get_privileged_observations()
        dones = terminated | truncated
        return obs, privileged_obs, rew, dones, extras
    
    def reset(self):
        obs_dict, _ = self.env.reset()
        obs = self._strip_tensordict(obs_dict["policy"])
        return obs, self.get_privileged_observations()

# --- [参数过滤器] ---
def filter_dict(raw_dict, whitelist):
    return {k: v for k, v in raw_dict.items() if k in whitelist}

PPO_WHITELIST = ['value_loss_coef', 'use_clipped_value_loss', 'clip_param', 'entropy_coef', 
                 'num_learning_epochs', 'num_mini_batches', 'learning_rate', 'schedule', 
                 'gamma', 'lam', 'desired_kl', 'max_grad_norm']
POLICY_WHITELIST = ['init_noise_std', 'actor_hidden_dims', 'critic_hidden_dims', 'activation']

# -------------------------------------------------------

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # 1. 同步参数
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    # === 地形与高度修正 ===
    if hasattr(env_cfg.scene, "terrain"):
        print("[INFO] Play模式: 强制地形为无限平面 (Plane)")
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None 
    
    if hasattr(env_cfg.scene, "robot"):
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0)
        print("[INFO] Play模式: 强制初始高度为 1.05m")

    # 2. 设置日志根目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    # 3. 寻找 Checkpoint
    load_run = agent_cfg.load_run
    if load_run == "-1":
        load_run = None
    
    resume_path = None
    try:
        resume_path = get_checkpoint_path(log_root_path, load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] 加载模型路径: {resume_path}")
    except Exception as e:
        print(f"[ERROR] 无法找到模型 checkpoint: {e}")
        simulation_app.close()
        sys.exit(1)

    # 4. 创建环境
    print(f"[INFO] 正在创建环境: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    env = RslRlVecEnvWrapperFixed(env)

    # 5. 构造 Runner 配置
    raw_dict = class_to_dict(agent_cfg) if not isinstance(agent_cfg, dict) else agent_cfg
    
    # === [关键修复] 补全 runner 所需的所有键值 ===
    rsl_cfg = {
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "experiment_name": agent_cfg.experiment_name,
            "checkpoint": resume_path,
            
            # --- 以下是本次修复补充的必要参数 ---
            "num_steps_per_env": agent_cfg.num_steps_per_env,  # 解决 KeyError
            "max_iterations": agent_cfg.max_iterations,        # 初始化需要
            "save_interval": agent_cfg.save_interval,          # 初始化需要
            "run_name": agent_cfg.run_name,
        },
        "algorithm": filter_dict(raw_dict.get("algorithm", {}), PPO_WHITELIST),
        "policy": filter_dict(raw_dict.get("policy", {}), POLICY_WHITELIST),
    }

    # 6. 初始化 Runner
    # log_dir=None 表示不创建新的日志文件夹
    runner = OnPolicyRunner(env, rsl_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    
    policy = runner.get_inference_policy(device=env.device)

    # 7. 推理循环
    print("-" * 80)
    print("[INFO] 启动成功！在 Isaac Sim 中按 'F' 键跟随机器人。")
    print("-" * 80)

    obs, _ = env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)

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