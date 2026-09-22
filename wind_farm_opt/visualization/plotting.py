"""可视化绘图模块。

使用 Matplotlib 实现各种可视化图表。
"""

import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import Normalize, LinearSegmentedColormap

from ..constraints.boundary import SiteBoundary
from ..core.wind_resource import WindResource
from ..farm.aep import FarmResult
from ..optimization.ga import OptimizeResult


def set_chinese_font() -> None:
    """设置中文字体支持。"""
    # 字体配置不能只写入一个可能不存在的字体名，否则无头 Linux
    # 环境会在每个图表元素上重复输出 findfont 警告。
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if font in installed:
            plt.rcParams["font.sans-serif"] = [font]
            break
    else:
        # DejaVu Sans 是 Matplotlib 的稳定后备字体，至少保证拉丁字符和
        # 负号正常显示；没有中文字体时不再反复尝试不存在的字体。
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_farm_layout(
    positions: np.ndarray,
    boundary: SiteBoundary,
    rotor_diameters: np.ndarray,
    turbine_losses: Optional[np.ndarray] = None,
    turbine_names: Optional[list[str]] = None,
    wake_interactions: Optional[dict] = None,
    title: str = "风电场机位布局",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制风电场机位布局俯视图。

    Parameters
    ----------
    positions : np.ndarray
        风机位置 (N, 2)
    boundary : SiteBoundary
        场地边界
    rotor_diameters : np.ndarray
        每台风机的转子直径 (N,)
    turbine_losses : Optional[np.ndarray]
        每台风机的尾流损失百分比 (N,)，用于着色
    turbine_names : Optional[list[str]]
        风机名称/编号
    wake_interactions : Optional[dict]
        尾流相互作用信息，用于绘制尾流连线
    title : str
        图表标题
    save_path : Optional[str]
        保存路径
    show : bool
        是否显示图表
    """
    set_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 8))

    poly = Polygon(
        boundary.vertices,
        facecolor="lightgreen",
        edgecolor="darkgreen",
        linewidth=2,
        alpha=0.3,
        label="场地边界",
    )
    ax.add_patch(poly)

    if turbine_losses is not None:
        norm = Normalize(vmin=0, vmax=max(30.0, np.max(turbine_losses)))
        cmap = plt.get_cmap("YlOrRd")

        for i, (pos, d, loss) in enumerate(zip(positions, rotor_diameters, turbine_losses)):
            color = cmap(norm(loss))
            circle = Circle(pos, d / 2.0, facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.8)
            ax.add_patch(circle)

            if turbine_names is not None:
                ax.text(
                    pos[0],
                    pos[1],
                    turbine_names[i],
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("尾流损失 (%)")
    else:
        for i, (pos, d) in enumerate(zip(positions, rotor_diameters)):
            circle = Circle(pos, d / 2.0, facecolor="steelblue", edgecolor="darkblue", linewidth=1.5, alpha=0.7)
            ax.add_patch(circle)

            if turbine_names is not None:
                ax.text(
                    pos[0],
                    pos[1],
                    turbine_names[i],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )

    if wake_interactions is not None:
        for (up_idx, down_idx), intensity in wake_interactions.items():
            if intensity > 0.01:
                ax.plot(
                    [positions[up_idx, 0], positions[down_idx, 0]],
                    [positions[up_idx, 1], positions[down_idx, 1]],
                    "r-",
                    alpha=min(0.8, intensity * 5),
                    linewidth=0.5 + intensity * 3,
                )

    margin = 0.1
    x_range = boundary.x_max - boundary.x_min
    y_range = boundary.y_max - boundary.y_min
    ax.set_xlim(
        boundary.x_min - margin * x_range,
        boundary.x_max + margin * x_range,
    )
    ax.set_ylim(
        boundary.y_min - margin * y_range,
        boundary.y_max + margin * y_range,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X 坐标 (m)")
    ax.set_ylabel("Y 坐标 (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"布局图已保存到: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_wind_rose(
    wind_resource: WindResource,
    title: str = "风玫瑰图",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制风玫瑰图。

    Parameters
    ----------
    wind_resource : WindResource
        风资源数据
    title : str
        图表标题
    save_path : Optional[str]
        保存路径
    show : bool
        是否显示图表
    """
    set_chinese_font()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")

    angles = np.deg2rad(90.0 - wind_resource.directions)
    frequencies = wind_resource.frequencies * 100
    speeds = wind_resource.mean_speeds

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=np.min(speeds), vmax=np.max(speeds))
    colors = cmap(norm(speeds))

    width = np.deg2rad(wind_resource.sector_widths[0]) * 0.9

    bars = ax.bar(
        angles,
        frequencies,
        width=width,
        bottom=0.0,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    for bar, freq, speed in zip(bars, frequencies, speeds):
        if freq > 1.0:
            angle = bar.get_x() + bar.get_width() / 2
            ax.text(
                angle,
                freq + 0.5,
                f"{freq:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.12)
    cbar.set_label("平均风速 (m/s)")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.set_ylabel("频率 (%)", labelpad=20)
    ax.set_title(title, y=1.1, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"风玫瑰图已保存到: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_convergence(
    optimize_result: OptimizeResult,
    baseline_aep: Optional[float] = None,
    title: str = "优化收敛曲线",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制优化收敛曲线。

    Parameters
    ----------
    optimize_result : OptimizeResult
        优化结果
    baseline_aep : Optional[float]
        基线布局AEP（用于对比）(MWh)
    title : str
        图表标题
    save_path : Optional[str]
        保存路径
    show : bool
        是否显示图表
    """
    set_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    generations = np.arange(1, len(optimize_result.convergence_history) + 1)

    best_history = np.array(optimize_result.convergence_history) / 1e3
    mean_history = np.array(optimize_result.mean_history) / 1e3

    ax.plot(
        generations,
        best_history,
        "b-",
        linewidth=2,
        label="最优个体",
    )
    ax.plot(
        generations,
        mean_history,
        "g--",
        linewidth=1.5,
        alpha=0.7,
        label="种群平均",
    )

    if baseline_aep is not None:
        ax.axhline(
            y=baseline_aep / 1e3,
            color="r",
            linestyle=":",
            linewidth=2,
            label=f"网格布局基线: {baseline_aep/1e3:.2f} GWh",
        )

    improvement = 0.0
    if baseline_aep is not None and baseline_aep > 0:
        improvement = (optimize_result.best_fitness - baseline_aep) / baseline_aep * 100

    ax.set_xlabel("迭代代数")
    ax.set_ylabel("净年发电量 (GWh)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    info_text = (
        f"最优解: {optimize_result.best_fitness/1e3:.2f} GWh\n"
        f"找到代数: {optimize_result.best_generation}"
    )
    if improvement > 0:
        info_text += f"\n相对提升: {improvement:.2f}%"

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"收敛曲线已保存到: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_aep_vs_turbines(
    n_turbines_list: list[int],
    aep_list: list[float],
    lcoe_list: Optional[list[float]] = None,
    title: str = "发电量/度电成本 vs 风机台数",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制发电量随风机台数变化的曲线。

    Parameters
    ----------
    n_turbines_list : list[int]
        风机台数列表
    aep_list : list[float]
        对应净AEP列表 (MWh/year)
    lcoe_list : Optional[list[float]]
        对应LCOE列表 (元/kWh)
    title : str
        图表标题
    save_path : Optional[str]
        保存路径
    show : bool
        是否显示图表
    """
    set_chinese_font()

    if lcoe_list is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))

    n_turbines = np.array(n_turbines_list)
    aep_gwh = np.array(aep_list) / 1e3

    ax1.plot(n_turbines, aep_gwh, "bo-", linewidth=2, markersize=8, label="净AEP")
    ax1.set_ylabel("净年发电量 (GWh)")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    for i, (n, aep) in enumerate(zip(n_turbines, aep_gwh)):
        ax1.annotate(
            f"{aep:.1f}",
            (n, aep),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    if lcoe_list is not None:
        lcoe_array = np.array(lcoe_list)
        ax2.plot(n_turbines, lcoe_array, "ro-", linewidth=2, markersize=8, label="LCOE")
        ax2.set_xlabel("风机台数")
        ax2.set_ylabel("度电成本 (元/kWh)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        for i, (n, lcoe) in enumerate(zip(n_turbines, lcoe_array)):
            ax2.annotate(
                f"{lcoe:.3f}",
                (n, lcoe),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )
    else:
        ax1.set_xlabel("风机台数")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"台数-发电量曲线已保存到: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_turbine_loss_bar(
    farm_result: FarmResult,
    title: str = "各风机尾流损失",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制各风机尾流损失柱状图。

    Parameters
    ----------
    farm_result : FarmResult
        风电场计算结果
    title : str
        图表标题
    save_path : Optional[str]
        保存路径
    show : bool
        是否显示图表
    """
    set_chinese_font()

    fig, ax = plt.subplots(figsize=(12, 6))

    n_turb = len(farm_result.turbine_results)
    indices = np.arange(n_turb)
    loss_pcts = [tr.wake_loss_pct for tr in farm_result.turbine_results]
    net_aeps = [tr.net_aep for tr in farm_result.turbine_results]

    bars = ax.bar(indices, loss_pcts, color="salmon", edgecolor="darkred", alpha=0.8)

    for i, (bar, loss) in enumerate(zip(bars, loss_pcts)):
        height = bar.get_height()
        tr = farm_result.turbine_results[i]
        dom_source = tr.dominant_wake_source
        label = f"{loss:.1f}%"
        if dom_source is not None:
            label += f"\n(#{dom_source})"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.3,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    avg_loss = np.mean(loss_pcts)
    ax.axhline(
        y=avg_loss,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"平均损失: {avg_loss:.1f}%",
    )

    ax.set_xlabel("风机编号")
    ax.set_ylabel("尾流损失 (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"#{i}" for i in indices], fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"风机损失柱状图已保存到: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_comparison(
    baseline_result: FarmResult,
    optimized_result: FarmResult,
    title: str = "优化前后对比",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制优化前后关键指标对比图。

    Parameters
    ----------
    baseline_result : FarmResult
        基线布局结果
    optimized_result : FarmResult
        优化后布局结果
    title : str
        图表标题
    save_path : Optional[str]
        保存路径
    show : bool
        是否显示图表
    """
    set_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ["净AEP", "尾流损失", "容量系数", "单机平均AEP"]
    baseline_vals = [
        baseline_result.net_aep,
        baseline_result.wake_loss_pct,
        baseline_result.capacity_factor,
        baseline_result.net_aep / len(baseline_result.turbine_results),
    ]
    optimized_vals = [
        optimized_result.net_aep,
        optimized_result.wake_loss_pct,
        optimized_result.capacity_factor,
        optimized_result.net_aep / len(optimized_result.turbine_results),
    ]
    units = ["GWh", "%", "%", "GWh/台"]

    for i, ax in enumerate(axes.flat):
        x = ["基线", "优化后"]
        y = [baseline_vals[i], optimized_vals[i]]
        colors = ["skyblue", "lightgreen"]

        bars = ax.bar(x, y, color=colors, edgecolor="black", linewidth=1.5, alpha=0.8)

        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f} {units[i]}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        improvement = 0.0
        if baseline_vals[i] > 0:
            if i == 1:
                improvement = (baseline_vals[i] - optimized_vals[i]) / baseline_vals[i] * 100
                ax.set_title(f"{metrics[i]} (减少 {improvement:.1f}%)", fontsize=12, fontweight="bold")
            else:
                improvement = (optimized_vals[i] - baseline_vals[i]) / baseline_vals[i] * 100
                ax.set_title(f"{metrics[i]} (提升 {improvement:.1f}%)", fontsize=12, fontweight="bold")
        else:
            ax.set_title(metrics[i], fontsize=12, fontweight="bold")

        ax.set_ylabel(units[i])
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"对比图已保存到: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_wake_heatmap(
    positions: np.ndarray,
    boundary: SiteBoundary,
    wake_model,
    wind_direction: float,
    rotor_diameters: np.ndarray,
    thrust_coefficients: np.ndarray,
    grid_resolution: int = 100,
    title: str = "尾流速度亏损分布",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制尾流速度亏损热力图（单一风向）。

    Parameters
    ----------
    positions : np.ndarray
        风机位置 (N, 2)
    boundary : SiteBoundary
        场地边界
    wake_model
        尾流模型实例
    wind_direction : float
        风向 (度)
    rotor_diameters : np.ndarray
        转子直径数组 (N,)
    thrust_coefficients : np.ndarray
        推力系数数组 (N,)
    grid_resolution : int
        网格分辨率
    title : str
        图表标题
    save_path : Optional[str]
        保存路径
    show : bool
        是否显示图表
    """
    set_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 8))

    x = np.linspace(boundary.x_min, boundary.x_max, grid_resolution)
    y = np.linspace(boundary.y_min, boundary.y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)

    wind_rad = np.deg2rad(270.0 - wind_direction)
    wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])

    deficit_field = np.zeros_like(X)

    n_turb = positions.shape[0]
    for i in range(n_turb):
        pos_i = positions[i]

        delta = np.stack([X - pos_i[0], Y - pos_i[1]], axis=-1)
        dist = np.linalg.norm(delta, axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            delta_norm = np.where(
                dist[..., np.newaxis] > 1e-12,
                delta / dist[..., np.newaxis],
                0.0,
            )

        along_wind = np.sum(delta_norm * wind_vec, axis=-1)
        downstream_mask = (along_wind > 0.0) & (dist > 1e-12)

        downstream_dist = np.where(downstream_mask, dist * along_wind, 0.0)
        cross_dist = np.where(
            downstream_mask,
            dist * np.sqrt(np.clip(1.0 - along_wind ** 2, 0.0, 1.0)),
            0.0,
        )

        wr = wake_model.wake_radius(downstream_dist, rotor_diameters[i])
        peak_def = wake_model.velocity_deficit(
            downstream_dist,
            rotor_diameters[i],
            thrust_coefficients[i],
        )
        radial = wake_model.radial_profile(cross_dist, wr)

        deficit_i = peak_def * radial
        deficit_i = np.where(downstream_mask, deficit_i, 0.0)

        deficit_field = np.sqrt(deficit_field ** 2 + deficit_i ** 2)

    deficit_field = np.clip(deficit_field, 0.0, 1.0)

    mask = np.zeros_like(deficit_field, dtype=bool)
    for xi in range(grid_resolution):
        for yi in range(grid_resolution):
            pt = np.array([X[yi, xi], Y[yi, xi]])
            mask[yi, xi] = not boundary.contains_point(pt)

    deficit_masked = np.ma.masked_where(mask, deficit_field)

    contour = ax.contourf(
        X, Y, deficit_masked * 100,
        levels=np.linspace(0, 50, 21),
        cmap="hot_r",
        alpha=0.7,
    )

    poly = Polygon(
        boundary.vertices,
        facecolor="none",
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(poly)

    for pos, d in zip(positions, rotor_diameters):
        circle = Circle(pos, d / 2.0, facecolor="white", edgecolor="blue", linewidth=2)
        ax.add_patch(circle)

    ax.quiver(
        boundary.x_max - 200,
        boundary.y_max - 200,
        wind_vec[0],
        wind_vec[1],
        scale=5,
        width=0.02,
        color="blue",
    )
    ax.text(
        boundary.x_max - 200,
        boundary.y_max - 400,
        f"风向 {wind_direction:.0f}°",
        ha="center",
        va="top",
        fontsize=10,
        color="blue",
    )

    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("速度亏损 (%)")

    ax.set_aspect("equal")
    ax.set_xlabel("X 坐标 (m)")
    ax.set_ylabel("Y 坐标 (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"尾流热力图已保存到: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
