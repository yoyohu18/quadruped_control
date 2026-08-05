"""观测归一化：running mean/std。

## 为什么必须做

策略网络的输入里同时有：

* 基座角速度，量级 ~1 rad/s；
* 投影重力，量级 ~1（单位向量）；
* 关节位置偏差，量级 ~0.1 rad；
* 关节速度，量级 ~10 rad/s；
* 速度指令，量级 ~1 m/s。

如果不归一化，MLP 第一层的梯度会被关节速度那一段主导，其余通道学不动。
这与无人机上把状态量做无量纲化（除以特征长度、特征时间）是同一件事，
只不过这里的"特征尺度"是从数据里在线估出来的，而不是手推的。

## 为什么是 Welford 而不是"存下所有数据再算"

在线学习里样本是流式的，几十亿步不可能存下来。Welford / Chan 的并行合并
公式允许**逐批**更新且数值稳定 —— 直接累加 :math:`\\sum x^2` 在
float32 下会灾难性抵消。

## 冻结的时机

评估（play）与部署时必须**冻结**统计量：继续更新会让同一段观测在不同时刻
被映射到不同的网络输入，策略行为随之漂移。这与状态估计里的在线标定必须在
飞行中冻结是同一个道理。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RunningMeanStd", "EmpiricalNormalization"]


class RunningMeanStd(nn.Module):
    """逐通道的均值/方差在线估计（Chan 并行合并公式）。

    统计量注册为 buffer 而非 parameter —— 它们随 checkpoint 保存，但**不参与
    梯度更新**。这一点很容易写错：如果误写成 parameter，优化器会去"优化"
    数据的均值，训练会以非常隐蔽的方式变坏。

    Args:
        shape: 单个样本的形状，例如 ``(obs_dim,)``。
        epsilon: 初始伪计数，避免第一批数据的方差为 0。
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """用一批样本更新统计量。

        Args:
            x: ``(batch, *shape)``。
        """
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total
        # Chan 合并：M2_total = M2_a + M2_b + delta^2 * n_a * n_b / n_total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)


class EmpiricalNormalization(nn.Module):
    """把观测映射到零均值单位方差，训练时在线更新统计量。

    Args:
        shape: 观测维度。
        epsilon: 除法保护项。
        until: 累计样本数超过它之后停止更新（``None`` 表示一直更新）。
            典型做法是让统计量在训练早期收敛后就基本不动，这样策略面对的
            输入分布是平稳的。
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-8, until: int | None = None) -> None:
        super().__init__()
        self.rms = RunningMeanStd(shape)
        self.epsilon = epsilon
        self.until = until

    @property
    def frozen(self) -> bool:
        return self.until is not None and float(self.rms.count) >= self.until

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.frozen:
            self.rms.update(x)
        return (x - self.rms.mean) / torch.sqrt(self.rms.var + self.epsilon)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """归一化的逆变换，调试时用来把网络输入还原成物理量。"""
        return y * torch.sqrt(self.rms.var + self.epsilon) + self.rms.mean
