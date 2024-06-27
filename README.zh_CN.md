# 通过强化学习优化游戏策略：贪吃蛇游戏的案例研究

<p align="center">
  <a href="README.md">English</a> •
  <a href="README.ja_JP.md">日本語 (Japanese)</a> •
  <a href="README.zh_CN.md">简体中文 (Simplified Chinese)</a> 
</p>

你现在所在的分支为 ddqn 分支！

## 概要

这项研究聚焦于强化学习在游戏策略开发中的应用，以经典的"贪吃蛇游戏"为案例。研究目标包括开发能有效玩游戏的AI代理，比较不同强化学习算法的效果，并探讨这些技术在更广泛AI领域的应用潜力。实验采用DQN、Double DQN和Dueling DQN三种算法，在模拟环境中训练AI代理，并设置多项性能指标。研究旨在深化对强化学习在游戏策略开发中应用的理解，为更复杂场景提供框架，并推动适应性更强的AI系统开发。

## 查看不同算法实现的代码

你可以通过切换分支来查看不同算法实现的代码：

- [main](https://github.com/chenxingxu3/snake-ai-pytorch/tree/main)：DQN 算法

- [ddqn](https://github.com/chenxingxu3/snake-ai-pytorch/tree/ddqn)：Double DQN 算法

- [DuelingDQN](https://github.com/chenxingxu3/snake-ai-pytorch/tree/DuelingDQN)：Dueling DQN 算法

## 在CONDA中创建运行环境

从文件创建环境`snake-ai-pytorch.yml`：

```shell
conda env create -f snake-ai-pytorch.yml
```

激活新环境：`conda activate snake-ai-pytorch`

验证新环境是否已正确安装：

```shell
conda env list
```

您也可以使用：`conda info --envs`

更多信息请访问：[CONDA User guide  > Managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## 在ANACONDA中创建运行环境【推荐】

ANACONDA下载地址：[https://www.anaconda.com/download](https://www.anaconda.com/download)

启动`ANACONDA.NAVIGATOR`，在`Environments`>`Import`中导入`snake-ai-pytorch.yml`

![](assets_README.zh_CN/2024-06-25-11-41-49-image.png)

## 运行游戏，并开始训练过程

使用Python运行`agent.py`：

```shell
python agent.py
```

游戏运行截图：

![](assets_README.zh_CN/2024-06-25-11-48-31-image.png)

## 训练结果

你可以在`results`目录下查看训练结果。

## 训练过程录像(1000回)

- DQN：[利用Deep Q Learning算法训练AI玩贪吃蛇游戏（1000回）](https://odysee.com/@Xingxu:4/snake-dqn-1000-episodes:5?r=3voigLSm5Gk2uFYiE7h2PoseeErFC63k)

- Double DQN：[利用Double DQN算法训练AI玩贪吃蛇游戏（1000回）](https://odysee.com/@Xingxu:4/snake-ddqn-1000-episodes:6?r=3voigLSm5Gk2uFYiE7h2PoseeErFC63k)

- Dueling DQN：[利用Dueling DQN算法训练AI玩贪吃蛇游戏（1000回）](https://odysee.com/@Xingxu:4/snake-duelingdqn-1000-episodes:3?r=3voigLSm5Gk2uFYiE7h2PoseeErFC63k)

## 参考的代码

贪吃蛇游戏的设计以及DQN算法的实现，参考了以下项目中的代码：

[GitHub - patrickloeber/snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch)

Double DQN和Dueling DQN算法的实现，参考了以下项目中的代码：

[GitHub - p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch: PyTorch implementations of deep reinforcement learning algorithms and environments](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
