强化学习 (Reinforcement Learning, RL) 是机器学习的核心分支之一，核心逻辑是**代理（Agent）**在**环境（Environment）**中通过 “**经验积累**” 学习**最优行为策略**：代理通过执行动作与环境交互，环境会反馈 “**奖励**”（正向反馈，如达成目标）或 “**惩罚**”（负向反馈，如失败），代理则以 “**最大化累积奖励**” 为目标，不断调整行为，最终学会在特定场景下的最优决策方式. 它与监督学习（依赖标注数据）、无监督学习（挖掘数据内在规律）的核心区别在于：**无预设 “正确答案”，仅通过环境反馈的 “奖励信号” 动态学习**，更贴近人类 / 动物从经验中学习的过程. 为了实现 RL, 我选择了两个 SDK: **Gymnasium** 和 **Stable Baselines3**. Gymnasium 是一款开源 Python 库，主要用于强化学习环境的开发与算法性能对比。它的核心功能包括两方面：一是提供一套标准的 RL 环境 API（应用程序编程接口），实现代理与环境之间的通信交互；二是提供一组符合该 API 规范的标准环境集合. 而 Stable Baselines3（SB3）是基于 PyTorch 开发的一套可靠的强化学习算法实现集合, 其实现了 `A2C` `DDPG` `TRPO` `PPO` `DQN` 等经典算法, 可开箱即用并用于代理 RL 训练. 结合 Gymnasium 实现 RL 环境定义, 并结合 Stable Baselines3 的预定义算法, 我们可以实现深度强化学习的训练与评测.

## 本次复现的算法 - PPO

PPO（Proximal Policy Optimization，近端策略优化）是一种基于策略优化的强化学习算法，由 OpenAI 于 2017 年提出$^{[1]}$。它旨在解决传统策略梯度方法中训练不稳定、效率低下的问题。PPO 通过引入一个截断的目标函数，限制新策略和旧策略之间的差异，避免策略更新过快导致的不稳定性，从而提高训练的稳定性和效率.

关于 PPO 算法原理的详细解释, 请查看[我的另一篇博客](https://vortezwohl.github.io/rl/2025/03/14/%E6%B7%B1%E5%85%A5%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E7%AC%94%E8%AE%B0.html#ppo).

## 实验设计

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

3. ### 开始 RL 训练

    ```python
    from stable_baselines3 import PPO

    env = VectorClassificationEnv(np.random.randn(1000, 128), np.random.randint(0, 2, 1000))
    ppo = PPO(MyPolicy, env, verbose=2, learning_rate=2e-6)
    ppo.learn(total_timesteps=3000, progress_bar=True)
    ```

## 参考文献

[[1](https://doi.org/10.48550/arXiv.1707.06347)] John Schulman et al. Proximal Policy Optimization Algorithms. *arXiv preprint*, 2017.

[[2](https://doi.org/10.48550/arXiv.2407.17032)] Towers et al. Gymnasium: A Standard Interface for Reinforcement Learning Environments. *arXiv preprint*, 2024.

[[3](http://jmlr.org/papers/v22/20-1364.html)] Antonin Raffin et al. Stable-Baselines3: Reliable Reinforcement Learning Implementations. *Journal of Machine Learning Research*, 2021.
