"""尾流模型实现。

包含 Jensen 模型和高斯剖面尾流模型，以及多尾流叠加（平方和）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class WakeModel(ABC):
    """尾流模型抽象基类。"""

    @abstractmethod
    def velocity_deficit(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
        thrust_coefficient: float,
    ) -> float | np.ndarray:
        """计算轴向速度亏损。

        Parameters
        ----------
        distance : float | np.ndarray
            下游距离 (m)
        rotor_diameter : float
            上游风机转子直径 (m)
        thrust_coefficient : float
            上游风机推力系数

        Returns
        -------
        float | np.ndarray
            速度亏损 (1 - u/U0)，范围 [0, 1)
        """
        pass

    @abstractmethod
    def wake_radius(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
    ) -> float | np.ndarray:
        """计算尾流半径。

        Parameters
        ----------
        distance : float | np.ndarray
            下游距离 (m)
        rotor_diameter : float
            上游风机转子直径 (m)

        Returns
        -------
        float | np.ndarray
            尾流半径 (m)
        """
        pass

    def radial_profile(
        self,
        radial_dist: float | np.ndarray,
        wake_radius: float,
    ) -> float | np.ndarray:
        """计算径向速度亏损分布。

        Parameters
        ----------
        radial_dist : float | np.ndarray
            到尾流中心线的径向距离 (m)
        wake_radius : float
            尾流半径 (m)

        Returns
        -------
        float | np.ndarray
            径向分布系数，范围 [0, 1]
        """
        pass


class JensenWake(WakeModel):
    """Jensen 尾流模型 (1983)。

    经典的锥形尾流模型，假设尾流线性扩张，速度亏损在尾流截面均匀分布。
    """

    def __init__(self, wake_decay: float = 0.07) -> None:
        self.wake_decay = wake_decay

    def velocity_deficit(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
        thrust_coefficient: float,
    ) -> float | np.ndarray:
        dist = np.asarray(distance, dtype=np.float64)
        d0 = rotor_diameter

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = d0 / (d0 + 2.0 * self.wake_decay * dist)
            deficit = 1.0 - np.sqrt(1.0 - thrust_coefficient) * ratio ** 2

        deficit = np.where(dist <= 0, 0.0, deficit)
        deficit = np.clip(deficit, 0.0, 1.0)

        return deficit if dist.ndim > 0 else float(deficit)

    def wake_radius(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
    ) -> float | np.ndarray:
        dist = np.asarray(distance, dtype=np.float64)
        radius = 0.5 * rotor_diameter + self.wake_decay * dist
        radius = np.where(dist <= 0, 0.5 * rotor_diameter, radius)
        return radius if dist.ndim > 0 else float(radius)

    def radial_profile(
        self,
        radial_dist: float | np.ndarray,
        wake_radius: float,
    ) -> float | np.ndarray:
        r = np.asarray(radial_dist, dtype=np.float64)
        profile = np.where(np.abs(r) <= wake_radius, 1.0, 0.0)
        return profile if r.ndim > 0 else float(profile)


class GaussianWake(WakeModel):
    """Bastankhah & Porté-Agel 高斯剖面尾流模型 (2014)。

    假设尾流速度亏损符合高斯分布，更符合实际风洞和实测数据。
    """

    def __init__(
        self,
        wake_decay: float = 0.035,
        near_wake_length: float = 3.0,
    ) -> None:
        self.wake_decay = wake_decay
        self.near_wake_length = near_wake_length

    def velocity_deficit(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
        thrust_coefficient: float,
    ) -> float | np.ndarray:
        dist = np.asarray(distance, dtype=np.float64)
        d0 = rotor_diameter
        ct = thrust_coefficient

        beta = 0.5 * (1.0 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct)

        with np.errstate(divide="ignore", invalid="ignore"):
            x_nd = dist / d0
            x_over = np.maximum(x_nd, self.near_wake_length)
            sigma = self.wake_decay * x_over * d0 + d0 / 2.0 / np.sqrt(beta)

            peak_deficit = (
                1.0
                - np.sqrt(
                    1.0
                    - ct
                    / (8.0 * (sigma / d0) ** 2)
                )
            )

        peak_deficit = np.where(dist <= 0, 0.0, peak_deficit)
        peak_deficit = np.clip(peak_deficit, 0.0, 1.0)

        return peak_deficit if dist.ndim > 0 else float(peak_deficit)

    def wake_radius(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
    ) -> float | np.ndarray:
        dist = np.asarray(distance, dtype=np.float64)
        d0 = rotor_diameter

        x_nd = dist / d0
        x_over = np.maximum(x_nd, self.near_wake_length)
        sigma = self.wake_decay * x_over * d0 + d0 / 4.0

        radius = 2.0 * sigma
        radius = np.where(dist <= 0, 0.5 * d0, radius)

        return radius if dist.ndim > 0 else float(radius)

    def radial_profile(
        self,
        radial_dist: float | np.ndarray,
        wake_radius: float,
    ) -> float | np.ndarray:
        r = np.asarray(radial_dist, dtype=np.float64)
        sigma = wake_radius / 2.0

        with np.errstate(divide="ignore", invalid="ignore"):
            profile = np.exp(-0.5 * (r / sigma) ** 2)

        profile = np.where(sigma <= 0, 0.0, profile)

        return profile if r.ndim > 0 else float(profile)


def superpose_wakes(
    deficits: np.ndarray,
    method: str = "sum_of_squares",
) -> np.ndarray:
    """叠加多个尾流的速度亏损。

    Parameters
    ----------
    deficits : np.ndarray
        每个上游风机产生的速度亏损数组，形状为 (N_upstream, ...)
    method : str
        叠加方法："sum_of_squares"（平方和，推荐）或 "linear"（线性叠加）

    Returns
    -------
    np.ndarray
        叠加后的总速度亏损，形状为 (...)
    """
    deficits = np.asarray(deficits, dtype=np.float64)

    if method == "sum_of_squares":
        total_deficit = np.sqrt(np.sum(deficits ** 2, axis=0))
    elif method == "linear":
        total_deficit = np.sum(deficits, axis=0)
    else:
        raise ValueError(f"未知的尾流叠加方法: {method}")

    return np.clip(total_deficit, 0.0, 1.0)


@dataclass
class WakeInteraction:
    """尾流相互作用结果。

    Parameters
    ----------
    upstream_idx : int
        上游风机索引
    downstream_idx : int
        下游风机索引
    distance : float
        两台风机之间的距离 (m)
    angle_from_wind : float
        两台风机连线与风向的夹角 (度)
    velocity_deficit : float
        速度亏损值
    affected_power : float
        受影响的功率损失 (kW)
    in_wake : bool
        是否在尾流影响范围内
    """

    upstream_idx: int
    downstream_idx: int
    distance: float
    angle_from_wind: float
    velocity_deficit: float
    affected_power: float
    in_wake: bool


def compute_wake_interactions(
    positions: np.ndarray,
    wind_direction: float,
    rotor_diameters: np.ndarray,
    thrust_coefficients: np.ndarray,
    wake_model: WakeModel,
    power_curves: list[np.ndarray],
    free_stream_speed: float,
) -> list[WakeInteraction]:
    """计算给定风向下的所有尾流相互作用。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N_turbines, 2)
    wind_direction : float
        风向 (度)，0度为北，顺时针
    rotor_diameters : np.ndarray
        每台风机的转子直径，形状为 (N_turbines,)
    thrust_coefficients : np.ndarray
        每台风机的推力系数，形状为 (N_turbines,)
    wake_model : WakeModel
        尾流模型实例
    power_curves : list[np.ndarray]
        每台风机的功率曲线
    free_stream_speed : float
        自由来流风速 (m/s)

    Returns
    -------
    list[WakeInteraction]
        尾流相互作用列表
    """
    n = positions.shape[0]
    interactions = []

    wind_rad = np.deg2rad(270.0 - wind_direction)
    wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            delta = positions[j] - positions[i]
            distance = np.linalg.norm(delta)

            if distance <= 0:
                continue

            delta_norm = delta / distance
            along_wind = np.dot(delta_norm, wind_vec)

            if along_wind <= 0:
                continue

            downstream_dist = distance * along_wind
            cross_dist = distance * np.sqrt(1.0 - along_wind ** 2)

            wr = wake_model.wake_radius(downstream_dist, rotor_diameters[i])
            radial_factor = wake_model.radial_profile(cross_dist, wr)
            peak_deficit = wake_model.velocity_deficit(
                downstream_dist, rotor_diameters[i], thrust_coefficients[i]
            )
            deficit = peak_deficit * radial_factor

            angle_deg = float(np.rad2deg(np.arccos(along_wind)))

            effective_speed = free_stream_speed * (1.0 - deficit)
            power_free = np.interp(free_stream_speed, power_curves[j][:, 0], power_curves[j][:, 1],
                                  left=0.0, right=0.0)
            power_wake = np.interp(effective_speed, power_curves[j][:, 0], power_curves[j][:, 1],
                                  left=0.0, right=0.0)
            power_loss = power_free - power_wake

            in_wake = radial_factor > 0.01 and deficit > 0.001

            interactions.append(
                WakeInteraction(
                    upstream_idx=i,
                    downstream_idx=j,
                    distance=float(distance),
                    angle_from_wind=angle_deg,
                    velocity_deficit=float(deficit),
                    affected_power=float(power_loss),
                    in_wake=bool(in_wake),
                )
            )

    return interactions
