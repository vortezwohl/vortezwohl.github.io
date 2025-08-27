---
layout: post
toc: true
title: "基于 Stable Baselines3 和 Gymnasium 的强化学习 (PPO) 算法实践"
categories: RL
tags: [RL, PPO, RLHF]
author:
  - vortezwohl
  - 吴子豪
---
强化学习 (Reinforcement Learning, RL) 是机器学习的核心分支之一，核心逻辑是**代理（Agent）**在**环境（Environment）**中通过 “**经验积累**” 学习**最优行为策略**：代理通过执行动作与环境交互，环境会反馈 “**奖励**”（正向反馈，如达成目标）或 “**惩罚**”（负向反馈，如失败），代理则以 “**最大化累积奖励**” 为目标，不断调整行为，最终学会在特定场景下的最优决策方式. 它与监督学习（依赖标注数据）、无监督学习（挖掘数据内在规律）的核心区别在于：**无预设 “正确答案”，仅通过环境反馈的 “奖励信号” 动态学习**，更贴近人类 / 动物从经验中学习的过程. 为了实现 RL, 我选择了两个 SDK: **Gymnasium** 和 **Stable Baselines3**. Gymnasium 是一款开源 Python 库，主要用于强化学习环境的开发与算法性能对比。它的核心功能包括两方面：一是提供一套标准的 RL 环境 API（应用程序编程接口），实现代理与环境之间的通信交互；二是提供一组符合该 API 规范的标准环境集合. 而 Stable Baselines3（SB3）是基于 PyTorch 开发的一套可靠的强化学习算法实现集合, 其实现了 `A2C` `DDPG` `TRPO` `PPO` `DQN` 等经典算法, 可开箱即用并用于代理 RL 训练. 结合 Gymnasium 实现 RL 环境定义, 并结合 Stable Baselines3 的预定义算法, 我们可以实现深度强化学习的训练与评测.

## 本次复现的算法 - PPO

PPO（Proximal Policy Optimization，近端策略优化）是一种基于策略优化的强化学习算法，由 OpenAI 于 2017 年提出$^{[1]}$。它旨在解决传统策略梯度方法中训练不稳定、效率低下的问题。PPO 通过引入一个截断的目标函数，限制新策略和旧策略之间的差异，避免策略更新过快导致的不稳定性，从而提高训练的稳定性和效率.

关于 PPO 算法原理的详细解释, 请查看[我的另一篇博客](https://vortezwohl.github.io/rl/2025/03/14/%E6%B7%B1%E5%85%A5%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E7%AC%94%E8%AE%B0.html#ppo).

## 实验设计

0. ### 安装依赖项

    SDK|Version|Repo
    |:-:|:-:|:-:|
    `gymnasium`|`1.2.0`|https://github.com/Farama-Foundation/Gymnasium.git
    `stable-baselines3`|`2.7.0`|https://github.com/DLR-RM/stable-baselines3.git

    ```
    uv add gymnasium==1.2.0
    ```
    ```
    uv add stable-baselines3==2.7.0
    ```

1. ### 定义环境

    > **环境 (Environment)**, 可类比为监督学习中的标注数据集, 监督学习基于数据集训练, 而强化学习 (特指在策略强化学习, 即 On-Policy RL) 基于实时环境训练. 对于离策略强化学习, 也可以认为其基于经验数据集训练, 本文主要介绍在策略强化学习.

    这里我定义简单的向量二分类环境:

    ```python
    import numpy as np
    from gymnasium import spaces, Env


    class VectorClassificationEnv(Env):  
        def __init__(self, features: np.ndarray, labels: np.ndarray):  
            super().__init__()  
            # 定义状态空间 (观察空间) 为无界连续值向量
            self.observation_space = spaces.Box(  
                low=-np.inf, high=np.inf,   
                shape=(features.shape[1],),   
                dtype=np.float32  
            )  
            # 定义动作空间为离散值标量
            self.action_space = spaces.Discrete(2)
            # 特征序列
            self.features = features
            # 标签序列
            self.labels = labels
            # 指针, 用于时间步计数
            self.ptr = 0
        
        def step(self, action: int):
            true_label = self.labels[self.ptr]  
            # 计算该动作下的奖励
            reward = 1.0 if action == true_label else -1.0  
            # 指针自增, 指向下一状态
            self.ptr += 1  
            terminated = self.ptr >= len(self.features)  
            if not terminated:  
                # 观察环境的当前状态
                observation = self.features[self.ptr]  
            else:  
                observation = np.zeros(self.observation_space.shape)  
            return observation, reward, terminated, False, {}  
        
        def reset(self, seed=None, options=None):  
            super().reset(seed=seed)  
            # 指针复位
            self.ptr = 0  
            observation = self.features[self.ptr]
            return observation, {}
    ```

    > **观察空间 (Observation)**, 代理在环境中所 "感知" 的信息 (观察值) 的范围和结构, 即代理接收的输入信息. 与之相关的另一个概念则是 **状态空间 (State Space)**, 状态空间即环境可能处于的所有可能状态的集合, 它描述了环境的完整信息, 在以上向量分类任务中, 状态空间被认为是等同于观察空间的, 因为该任务中, 代理能够完全观察到环境的所有状态信息, 是**完全可观察环境 (Fully Observable Environment)**, 但在某些任务中, 环境是**部分可观察的 (Partially Observable**), 例如机器人仅通过摄像头观察物理世界, 此时状态空间不等同于观察空间 (观察空间则是状态空间的一部分或有噪声的投影, 这种任务可能需要通过历史观察推断真实状态).

    > **动作空间 (Action Space)**, 代理在环境中可执行的动作集合, 即代理的输出. 在该环境中, 动作空间即 0 或 1, 代表正负两个类别.

    相关 `gymnasium` API 解释:

    - **Env**: `Env` 是 `gymnasium` 定义的标准环境接口, 通过实现该接口的 `step()` 方法, 可实现环境的单时间步交互, 而实现 `reset()` 方法则可以实现环境的状态重置.

    - **spaces**: `Space` 是 `gymnasium` 定义的标准值域接口, 其中最常用的 `Space` 子类有 `Box` 和 `Discrete` 两种.

        - **Box**: `Box` 表示连续或离散的 n 维数组, 支持有界和无界区间.

            其核心属性有 `low`(每个元素的最小值) `high`(每个元素的最大值) `shape`(数组形状) `dtype`(数据类型).

        - **Discrete**: `Discrete` 表示有限的整数集.

            其核心属性有 `n`(可能的值数量) `start`(起始值) `dtype`(数据类型)

        > 除了 `Box` 和 `Discrete`, `gymnasium` 还提供复合值域, 包括 `Dict` `Tuple` `MultiDiscrete` `MultiBinary` 等$^{[2]}$.

2. ### 定义策略

    对于 PPO 算法 (典型 ActorCritic 架构), 策略主要由策略网络 (`policy_net`) 和价值网络 (`value_net`) 构成, 以下是一个基于多头 FFN 的策略:

    ```python
    import torch
    from torch import nn
    from deeplotx import MultiHeadFeedForward
    from stable_baselines3.common.policies import ActorCriticPolicy


    class MyActorCritic(nn.Module):
        def __init__(self, feature_dim: int, policy_output_dim: int, value_output_dim: int, device: str = 'cpu', dtype: torch.dtype = torch.float32):
            super().__init__()  
            self.latent_dim_pi = policy_output_dim  
            self.latent_dim_vf = value_output_dim  
            # 创建策略网络
            self.policy_net = nn.Sequential(  
                # 网络主体
                MultiHeadFeedForward(feature_dim=feature_dim, num_heads=50, device=device, dtype=dtype), # 维度对齐
                # 输出维度对齐
                nn.Linear(in_features=feature_dim, out_features=policy_output_dim, device=torch.device(device), dtype=dtype)
            )  
            # 创建价值网络
            self.value_net = nn.Sequential(  
                # 网络主体
                MultiHeadFeedForward(feature_dim=feature_dim, num_heads=50, device=device, dtype=dtype), 
                # 输出维度对齐
                nn.Linear(in_features=feature_dim, out_features=value_output_dim, device=torch.device(device), dtype=dtype)
            )
        
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.policy_net.forward(x), self.value_net.forward(x)
        
        # 策略网络前向传播, 即执行动作
        def forward_actor(self, x: torch.Tensor):  
            return self.policy_net.forward(x)  

        # 价值网络前向传播, 即评估动作的价值
        def forward_critic(self, x: torch.Tensor):  
            return self.value_net.forward(x)


    class MyPolicy(ActorCriticPolicy):
        # 实现 ActorCriticPolicy 的 _build_mlp_extractor() 方法
        def _build_mlp_extractor(self) -> None:  
            # 在 _build_mlp_extractor() 方法中, 将 self.mlp_extractor 设置为自定义 ActorCritic 模型
            self.mlp_extractor = MyActorCritic(self.features_dim, 64, 64)
    ```

3. ### 策略优化

    以下代码基于上文定义的环境 (`VectorClassificationEnv`), 对上文定义的策略 (`MyPolicy`) 进行 PPO 优化 (训练):

    ```python
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback  

    # 创建 RL 环境
    env = VectorClassificationEnv(np.random.randn(1000, 128), np.random.randint(0, 2, 1000))

    # 定义 PPO 算法实现
    ppo = PPO(  
        policy=MyPolicy,                            # 策略网络类型
        env=env,                                    # 环境实例
        learning_rate=2e-6,                         # 学习率  
        n_steps=2048,                               # 单个 rollout 的采样时间步
        batch_size=64,                              # 批次大小  
        n_epochs=10,                                # 在单个 rollout buffer 上的训练轮数
        gamma=0.99,                                 # 折扣因子
        gae_lambda=0.95,                            # GAE lambda 参数  
        clip_range=0.2,                             # PPO 裁剪范围  
        clip_range_vf=None,                         # 价值函数裁剪范围  
        normalize_advantage=True,                   # 是否标准化优势  
        ent_coef=0.0,                               # 熵系数
        vf_coef=0.5,                                # 价值函数系数
        max_grad_norm=0.5,                          # 梯度裁剪最大范数
        use_sde=False,                              # 是否使用状态依赖探索 (SDE)
        sde_sample_freq=-1,                         # SDE采样频率
        rollout_buffer_class=None,                  # rollout 缓冲区类
        rollout_buffer_kwargs=None,                 # rollout 缓冲区参数  
        target_kl=None,                             # 目标 KL 散度  
        stats_window_size=100,                      # 统计窗口大小  
        tensorboard_log=None,                       # TensorBoard 日志路径, None 表示不记录日志  
        policy_kwargs=None,                         # 策略额外参数  
        verbose=2,                                  # 日志详细程度  
        seed=None,                                  # 随机种子  
        device="auto",                              # 计算设备  
        _init_setup_model=True                      # 是否初始化模型  
    )

    # 创建训练过程回调函数
    eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=500)  
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')

    # 开始训练
    ppo.learn(  
        total_timesteps=50000,  
        callback=[eval_callback, checkpoint_callback],  
        log_interval=10,  
        tb_log_name="ppo_run",  
        progress_bar=True  
    )
    ```

    `stable_baselines3.PPO` 参数解释:

    - **PPO**: `PPO` 是 `stable_baselines3` 中提供的 PPO 算法实现, 其源代码来源于 OpenAI 的 Paper 附件$^{[1]}$ (https://github.com/openai/spinningup/)

        - `policy`: 策略网络类型 (注意是类而不是示例)

        - `env`: 环境实例 (`gymnasium.Env` 实例)

        - `learning_rate`: 用于策略梯度更新的学习率

        - `n_steps`: 单个 rollout 所需的采样时间步

            > **Rollout (轨迹采样/推演)** 是强化学习核心概念之一, 其表示代理 (Agent) 在环境 (Env) 中遵循特定策略 (Policy) 执行动作, 得到一条完整的轨迹 (Trajectory, 即 状态-动作-奖励 序列) 的过程.

        - `batch_size`: 每次梯度更新时所使用的批次大小, 一个 batch 是从 rollout buffer 中采样得到的, 包含观察(状态) 动作 奖励 优势等数据, 一个合法的 `batch_size` 必须是 `n_step * n_envs` (即 rollout 的总大小) 的因子, 以确保能够完整利用所有数据.

            > `n_envs` 是并行运行的环境实例数量, 在这里为 1.

        - `gamma`: PPO 算法所涉及的 $\gamma$ 折扣因子$^{[1]}$

        - `gae_lambda`: GAE 中的 $\lambda$ 系数$^{[4]}$

        - `clip_range`: PPO 策略裁剪范围, 通过控制 PPO 策略的裁剪损失, 防止新策略和旧策略之间的偏离过大

        - `clip_range_vf`: 价值函数裁剪范围, 通过控制价值函数预测值的裁剪, 防止价值函数的更新过大

        - `normalize_advantage`: 是否对优势进行标准化

        - `ent_coef`: 熵系数, 这是在策略强化学习算法中用于控制探索与利用平衡的重要参数, 在 PPO 和 A2C 等在策略算法中，`ent_coef` 用于损失函数中作为熵损失项的权重系数.

        - `vf_coef`: 价值函数系数


            以上 `ent_coef` 和 `vf_coef` 的具体意义, 需结合下式解释:

            $$
            L^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]
            $$

            其中, $L^{CLIP}(\theta)$ 是策略损失, 用于限制策略的更新幅度, 避免过大的策略偏移, 而 $L^{VF}(\theta)$ 是价值函数损失, 用于优化状态价值函数 $V_{\theta}(s_t)$ 使其逼近实际奖励, $S[\pi_{\theta}](s_t)$ 则是策略的熵损失项, 用于鼓励探索 (熵越大, 策略越随机). 而 $c_1$ 和 $c_2$ 分别对应 `vf_coef` 和 `ent_coef`$^{[1]}$.

        - `max_grad_norm`: 梯度裁剪最大范数

        - `use_sde`: 是否使用状态依赖探索
            
            > **SDE (State-Dependent Exploration，状态依赖探索)** 是一种智能体的探索策略，其核心思想是：智能体的探索行为（如动作选择的随机性）会根据当前环境状态动态调整，而非采用固定不变的探索方式.

        - `sde_sample_freq`: 状态依赖探索采样频率, 该参数与状态依赖探索相关, 其决定了何时重新采样探索噪声矩阵. 设为 -1 (默认值) 时, 只在 rollout 开始时采样一次噪声矩阵，整个 rollout 过程中保持不变; 设为任意正数时, 每隔指定时间步重新采样一次噪声矩阵，例如设置为 4 表示每 4 步重新采样一次.

        - `rollout_buffer_class`: rollout 缓冲区类型, 对于默认情况, `stable_baseline3` 会自动选择合适的 buffer 类型.

            > 如果观察空间是 `spaces.Dict` 类型, 则使用 `DictRolloutBuffer`, 否则使用标准 `RolloutBuffer`, 当然也可以自定义 buffer.

        - `rollout_buffer_kwargs`: rollout 缓冲区参数, 传递给 `RolloutBuffer` 的构造函数的参数

        - `target_kl`: 目标 KL 散度

        - `stats_window_size`: 统计窗口大小, 其定义了用于计算滚动统计信息的窗口大小，具体指定了用于平均化报告的成功率、平均 episode 长度和平均奖励的 episode 数量.

            > 在训练日志中看到的 `rollout/ep_len_mean` `rollout/ep_rew_mean` 和 `rollout/success_rate` 等指标都是基于最近的 `stats_window_size` 个 episode 的平均值.

            > **episode (回合)** 指的是代理与回合制环境 (Episodic Environment) 从初始状态开始交互, 直到终止条件的完整交互过程.

        - `tensorboard_log`: tensorboard 日志路径

        - `policy_kwargs`: 策略额外参数

        - `verbose`: 日志详细程度

        - `seed`: 随机数种子

        - `device` 设备, 即 `pytorch` 设备

        - `_init_setup_model` 是否初始化模型, 当为 True (默认值) 时，会调用 `_setup_model()` 方法来初始化策略网络和 rollout buffer

    - **ppo.learn**: 

        以下只列举一些核心参数:

        - `total_timesteps`: 指定了训练的总环境交互时间步, 这是一个下界，实际训练步数可能会略微超过这个值，因为 PPO 会收集完整的 rollout.

        - `callback`: 传入了一个回调函数列表, 这些回调会在训练过程中的特定时刻被调用. 常用的回调函数包括 `eval_callback` 和 `checkpoint_callback`, 分别用于定期评估模型性能和定期保存模型检查点

        - `log_interval`: 日志记录间隔, 用于控制多少次训练迭代 (即策略更新) 后记录一次日志

4. ### 观察优化/训练日志

    ...待续...

## 参考文献

[[1](https://doi.org/10.48550/arXiv.1707.06347)] John Schulman et al. Proximal Policy Optimization Algorithms. *arXiv preprint*, 2017.

[[2](https://doi.org/10.48550/arXiv.2407.17032)] Towers et al. Gymnasium: A Standard Interface for Reinforcement Learning Environments. *arXiv preprint*, 2024.

[[3](http://jmlr.org/papers/v22/20-1364.html)] Antonin Raffin et al. Stable-Baselines3: Reliable Reinforcement Learning Implementations. *Journal of Machine Learning Research*, 2021.

[[4](https://doi.org/10.48550/arXiv.1506.02438)] John Schulman et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. *arXiv preprint*, 2015.
