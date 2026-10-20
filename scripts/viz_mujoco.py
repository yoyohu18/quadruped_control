"""里程碑 11 结果图：读取 ``logs/m11_mujoco_eval.json``，画四个面板。

先跑 ``python scripts/eval_mujoco.py``。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    R = json.loads((ROOT / "logs" / "m11_mujoco_eval.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))

    # ① 速度时间序列
    ax = axes[0, 0]
    for r, c in zip([r for r in R["speed"] if r["inertia"] == "urdf" and r["vx"] in (0.5, 1.0, 1.5)],
                    (BLUE, ORANGE, AQUA)):
        if r["fell"]:
            continue
        t, v = np.array(r["trace"]["t"]), np.array(r["trace"]["vx"])
        ax.plot(t, v, color=c, lw=1.5)
        ax.plot([0, 0.5, 0.5, t[-1]], [0, 0, r["vx"], r["vx"]], color=c, lw=1, ls="--")
        ax.text(t[-1] + 0.05, r["mean_vx"], f"{r['vx']} m/s", color=INK, fontsize=8, va="center")
    ax.set_xlabel("时间 [s]", color=INK2)
    ax.set_ylabel("前向速度 [m/s]", color=INK2)
    ax.set_title("① MuJoCo 闭环：指令（虚线）与实测（实线）", fontsize=11, color=INK)
    style(ax)

    # ② 指令 vs 稳态均速，两种模型参数
    ax = axes[0, 1]
    for inertia, c, lab in (("urdf", BLUE, "惯量与控制器一致"), ("mjcf", ORANGE, "官方 MJCF（轻 5.5%）")):
        rs = [r for r in R["speed"] if r["inertia"] == inertia and not r["fell"]]
        ax.errorbar([r["vx"] for r in rs], [r["mean_vx"] for r in rs], yerr=[r["rms"] for r in rs],
                    color=c, marker="o", ms=5, lw=1.5, capsize=3, label=lab)
        fell = [r for r in R["speed"] if r["inertia"] == inertia and r["fell"]]
        for r in fell:
            ax.plot(r["vx"], 0.05, marker="x", color=c, ms=9, mew=2)
    lim = max(r["vx"] for r in R["speed"])
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("指令速度 [m/s]", color=INK2)
    ax.set_ylabel("稳态均速 [m/s]（误差棒 = 速度 RMS）", color=INK2)
    ax.set_title("② 速度跟踪；× = 摔倒", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    # ③ 抗推：冲量 × 相位存活矩阵
    ax = axes[1, 0]
    phases = sorted({r["phase"] for r in R["push"]})
    impulses = sorted({r["impulse"] for r in R["push"]})
    for j, (d, lab) in enumerate((("lateral", "侧向"), ("forward", "前向"))):
        for i, p in enumerate(phases):
            y = j * (len(phases) + 1) + i
            for J in impulses:
                r = next(r for r in R["push"] if r["direction"] == d and r["phase"] == p and r["impulse"] == J)
                ax.scatter(J, y, s=70, marker="s", color=GRID if r["fell"] else BLUE,
                           edgecolor="white", linewidth=2)
            ax.text(impulses[0] - 2.2, y, f"{lab} · 相位 {p / 0.4:.2f}", fontsize=8, color=INK2, ha="right",
                    va="center")
        best = R["push_summary"][d]
        y0 = j * (len(phases) + 1)
        ax.vlines(best + 2.5, y0 - 0.5, y0 + len(phases) - 0.5, color=ORANGE, lw=1.2, ls="--")
        ax.text(best + 3.0, y0 + len(phases) - 0.35, f"{lab}：各相位都能恢复 ≤ {best} N·s",
                fontsize=8, color=INK, va="bottom")
    ax.set_yticks([])
    ax.set_ylim(-0.8, 2 * len(phases) + 1.2)
    ax.set_xlim(impulses[0] - 14, impulses[-1] + 4)
    ax.set_xlabel("冲量 [N·s]（0.1 s 推力，0.5 m/s 行走）", color=INK2)
    ax.set_title("③ 抗推：蓝=恢复，灰=摔倒", fontsize=11, color=INK)
    style(ax)
    ax.grid(False)

    # ④ 足端力估计
    ax = axes[1, 1]
    tr = next(r["trace"] for r in R["estimation"] if r["trace"])
    e = next(r for r in R["estimation"] if r["trace"])
    t = np.array(tr["t"]) - tr["t"][0]
    ax.plot(t, tr["truth"], color=GRID, lw=1.2, label="MuJoCo 真值（原始，含触地冲击）")
    ax.plot(t, tr["lowpass"], color=BLUE, lw=1.5, label="真值经同带宽低通")
    ax.plot(t, tr["est"], color=ORANGE, lw=1.5, ls="--", label="动量观测器估计")
    ax.set_xlabel("时间 [s]", color=INK2)
    ax.set_ylabel("左前脚法向力 [N]", color=INK2)
    ax.set_title(f"④ 足端力估计：vs 低通真值 RMS {e['rms_vs_lowpass']:.1f} N，"
                 f"接触检测 {e['detect_accuracy'] * 100:.1f}%", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    style(ax)

    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "m11_mujoco.png"
    fig.savefig(out, dpi=140, facecolor="white")
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
