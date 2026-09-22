"""风电场年发电量(AEP)计算模块。

高效的向量化尾流计算，支持多风向扇区和威布尔分布积分。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.turbine import Turbine
from ..core.wind_resource import WindResource
from ..core.wake import WakeModel, superpose_wakes


@dataclass
class TurbineResult:
    """单台风机的计算结果。

    Parameters
    ----------
    turbine_idx : int
        风机索引
    name : str
        风机名称
    gross_aep : float
        理论年发电量（无尾流）(MWh/year)
    net_aep : float
        净年发电量（考虑尾流）(MWh/year)
    wake_loss : float
        尾流损失 (MWh/year)
    wake_loss_pct : float
        尾流损失百分比 (%)
    capacity_factor : float
        容量系数 (%)
    avg_effective_speed : float
        平均有效风速 (m/s)
    dominant_wake_source : Optional[int]
        主要尾流来源风机索引
    total_power_loss_by_source : dict[int, float]
        各来源风机造成的损失
    """

    turbine_idx: int
    name: str
    gross_aep: float
    net_aep: float
    wake_loss: float
    wake_loss_pct: float
    capacity_factor: float
    avg_effective_speed: float
    dominant_wake_source: Optional[int]
    total_power_loss_by_source: dict[int, float] = field(default_factory=dict)


@dataclass
class FarmResult:
    """全场计算结果。

    Parameters
    ----------
    gross_aep : float
        理论年发电量（无尾流）(MWh/year)
    net_aep : float
        净年发电量（考虑尾流）(MWh/year)
    total_wake_loss : float
        总尾流损失 (MWh/year)
    wake_loss_pct : float
        尾流损失百分比 (%)
    capacity_factor : float
        容量系数 (%)
    total_installed_capacity : float
        总装机容量 (MW)
    turbine_results : list[TurbineResult]
        每台风机的详细结果
    sector_results : dict[int, dict]
        每个扇区的详细结果
    """

    gross_aep: float
    net_aep: float
    total_wake_loss: float
    wake_loss_pct: float
    capacity_factor: float
    total_installed_capacity: float
    turbine_results: list[TurbineResult]
    sector_results: dict[int, dict]


class AEPCalculator:
    """AEP计算器。

    使用向量化计算，提高效率。
    """

    def __init__(
        self,
        turbines: list[Turbine],
        wind_resource: WindResource,
        wake_model: WakeModel,
        wake_superposition: str = "sum_of_squares",
        speed_step: float = 0.5,
        speed_max: float = 30.0,
    ) -> None:
        """
        Parameters
        ----------
        turbines : list[Turbine]
            风机列表
        wind_resource : WindResource
            风资源数据
        wake_model : WakeModel
            尾流模型
        wake_superposition : str
            尾流叠加方法
        speed_step : float
            风速积分步长 (m/s)
        speed_max : float
            最大积分风速 (m/s)
        """
        self.turbines = turbines
        self.wind_resource = wind_resource
        self.wake_model = wake_model
        self.wake_superposition = wake_superposition
        self.speed_step = speed_step
        self.speed_max = speed_max

        self._speed_bins = np.arange(0.0, speed_max + speed_step, speed_step)
        self._speed_centers = self._speed_bins[:-1] + 0.5 * speed_step

        self._turbine_names = [t.name for t in turbines]
        self._rotor_diameters = np.array([t.rotor_diameter for t in turbines], dtype=np.float64)
        self._thrust_coefficients = np.array([t.thrust_coefficient for t in turbines], dtype=np.float64)
        self._rated_powers = np.array([t.rated_power for t in turbines], dtype=np.float64)
        self._power_curves = [t.power_curve for t in turbines]
        self._hub_heights = np.array([t.hub_height for t in turbines], dtype=np.float64)

        self._precompute_power_lookups()

    def _precompute_power_lookups(self) -> None:
        """预计算每台风机的功率查找表。"""
        n_turb = len(self.turbines)
        n_speed = len(self._speed_centers)

        self._power_lookup = np.zeros((n_turb, n_speed), dtype=np.float64)

        for i, turb in enumerate(self.turbines):
            self._power_lookup[i] = turb.power(self._speed_centers)

    def _compute_wake_deficit_field(
        self,
        positions: np.ndarray,
        wind_direction: float,
    ) -> np.ndarray:
        """计算给定风向下，每台风机在每个风速bin处的速度亏损。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)
        wind_direction : float
            风向 (度)

        Returns
        -------
        np.ndarray
            速度亏损数组 (N_turb,)，每个元素为该风机在该风向下的等效速度亏损
        """
        n = positions.shape[0]

        wind_rad = np.deg2rad(270.0 - wind_direction)
        wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])

        delta = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        distances = np.linalg.norm(delta, axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            delta_norm = np.where(
                distances[..., np.newaxis] > 1e-12,
                delta / distances[..., np.newaxis],
                0.0,
            )

        along_wind = np.sum(delta_norm * wind_vec, axis=-1)

        downstream_mask = (along_wind > 0.0) & (distances > 1e-12)

        downstream_dist = np.where(downstream_mask, distances * along_wind, 0.0)
        cross_dist = np.where(
            downstream_mask,
            distances * np.sqrt(np.clip(1.0 - along_wind ** 2, 0.0, 1.0)),
            0.0,
        )

        wr = self.wake_model.wake_radius(
            downstream_dist,
            self._rotor_diameters[:, np.newaxis],
        )

        peak_deficit = self.wake_model.velocity_deficit(
            downstream_dist,
            self._rotor_diameters[:, np.newaxis],
            self._thrust_coefficients[:, np.newaxis],
        )

        radial_factor = self.wake_model.radial_profile(cross_dist, wr)
        deficit_matrix = peak_deficit * radial_factor
        deficit_matrix = np.where(downstream_mask, deficit_matrix, 0.0)

        total_deficit = superpose_wakes(
            deficit_matrix,
            method=self.wake_superposition,
        )

        return total_deficit

    def _compute_sector_aep(
        self,
        positions: np.ndarray,
        sector_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算单个风向扇区的发电量。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)
        sector_idx : int
            扇区索引

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - 每台风机的净发电量 (N_turb,)
            - 每台风机的理论发电量 (N_turb,)
            - 每台风机的功率损失来源矩阵 (N_turb, N_turb)
        """
        sector = self.wind_resource.sectors[sector_idx]
        freq = sector.frequency
        wind_dir = sector.direction_center
        c = sector.weibull_c
        k = sector.weibull_k

        pdf = self.wind_resource.weibull_pdf(self._speed_centers, sector_idx)
        prob = pdf * self.speed_step

        total_deficit = self._compute_wake_deficit_field(positions, wind_dir)

        n_turb = len(self.turbines)
        n_speed = len(self._speed_centers)

        effective_speeds = self._speed_centers[np.newaxis, :] * (1.0 - total_deficit[:, np.newaxis])

        net_power = np.zeros((n_turb, n_speed))
        for i in range(n_turb):
            net_power[i] = np.interp(
                effective_speeds[i],
                self._power_curves[i][:, 0],
                self._power_curves[i][:, 1],
                left=0.0,
                right=0.0,
            )

        gross_power = self._power_lookup

        hours_per_year = 8760.0
        weighting = freq * hours_per_year * prob

        gross_aep_sector = np.sum(gross_power * weighting, axis=1)
        net_aep_sector = np.sum(net_power * weighting, axis=1)

        loss_by_source = self._compute_loss_by_source(
            positions,
            wind_dir,
            freq,
            hours_per_year,
            prob,
        )

        return net_aep_sector, gross_aep_sector, loss_by_source

    def _compute_loss_by_source(
        self,
        positions: np.ndarray,
        wind_direction: float,
        frequency: float,
        hours_per_year: float,
        prob: np.ndarray,
    ) -> np.ndarray:
        """计算每对风机之间的尾流能量损失。

        Returns
        -------
        np.ndarray
            损失矩阵 (N_turb, N_turb)，元素 [j, i] 表示风机 i 对 j 造成的损失
        """
        n = positions.shape[0]
        n_speed = len(self._speed_centers)

        wind_rad = np.deg2rad(270.0 - wind_direction)
        wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])

        loss_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                delta = positions[j] - positions[i]
                dist = np.linalg.norm(delta)
                if dist <= 1e-12:
                    continue

                delta_norm = delta / dist
                along_wind = np.dot(delta_norm, wind_vec)

                if along_wind <= 0:
                    continue

                downstream_dist = dist * along_wind
                cross_dist = dist * np.sqrt(np.clip(1.0 - along_wind ** 2, 0.0, None))

                wr = self.wake_model.wake_radius(downstream_dist, self._rotor_diameters[i])
                peak_def = self.wake_model.velocity_deficit(
                    downstream_dist,
                    self._rotor_diameters[i],
                    self._thrust_coefficients[i],
                )
                radial_factor = self.wake_model.radial_profile(cross_dist, wr)
                deficit_i_on_j = peak_def * radial_factor

                if deficit_i_on_j <= 0.001:
                    continue

                effective_speeds = self._speed_centers * (1.0 - deficit_i_on_j)
                power_def = np.interp(
                    effective_speeds,
                    self._power_curves[j][:, 0],
                    self._power_curves[j][:, 1],
                    left=0.0,
                    right=0.0,
                )
                power_gross = self._power_lookup[j]

                loss_ij = np.sum((power_gross - power_def) * frequency * hours_per_year * prob)

                loss_matrix[j, i] = float(loss_ij)

        return loss_matrix

    def compute_farm_aep(
        self,
        positions: np.ndarray,
        return_details: bool = True,
    ) -> FarmResult:
        """计算全场发电量。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)
        return_details : bool
            是否返回详细结果

        Returns
        -------
        FarmResult
            全场计算结果
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[0] != len(self.turbines):
            raise ValueError(
                f"位置数组形状应为 ({len(self.turbines)}, 2)，实际为 {positions.shape}"
            )

        n_turb = len(self.turbines)

        gross_aep_by_turbine = np.zeros(n_turb, dtype=np.float64)
        net_aep_by_turbine = np.zeros(n_turb, dtype=np.float64)
        loss_by_source_total = np.zeros((n_turb, n_turb), dtype=np.float64)

        sector_results = {}

        for s_idx in range(self.wind_resource.num_sectors):
            net_sector, gross_sector, loss_sector = self._compute_sector_aep(positions, s_idx)
            net_aep_by_turbine += net_sector
            gross_aep_by_turbine += gross_sector
            loss_by_source_total += loss_sector

            sector_results[s_idx] = {
                "direction": self.wind_resource.sectors[s_idx].direction_center,
                "frequency": self.wind_resource.sectors[s_idx].frequency,
                "net_aep": float(np.sum(net_sector)) / 1e3,
                "gross_aep": float(np.sum(gross_sector)) / 1e3,
            }

        gross_aep = float(np.sum(gross_aep_by_turbine)) / 1e3
        net_aep = float(np.sum(net_aep_by_turbine)) / 1e3
        total_loss = gross_aep - net_aep
        wake_loss_pct = (total_loss / gross_aep * 100.0) if gross_aep > 0 else 0.0

        total_installed = float(np.sum(self._rated_powers)) / 1e3
        capacity_factor = (net_aep / (total_installed * 8760.0) * 100.0) if total_installed > 0 else 0.0

        turbine_results = []
        for i in range(n_turb):
            gross = gross_aep_by_turbine[i] / 1e3
            net = net_aep_by_turbine[i] / 1e3
            loss = gross - net
            loss_pct = (loss / gross * 100.0) if gross > 0 else 0.0

            loss_sources = {}
            for j in range(n_turb):
                if i != j and loss_by_source_total[i, j] > 0.0:
                    loss_sources[j] = float(loss_by_source_total[i, j]) / 1e3

            dominant_source = None
            if loss_sources:
                dominant_source = max(loss_sources, key=loss_sources.get)

            turb_result = TurbineResult(
                turbine_idx=i,
                name=self._turbine_names[i],
                gross_aep=float(gross),
                net_aep=float(net),
                wake_loss=float(loss),
                wake_loss_pct=float(loss_pct),
                capacity_factor=float(net / (self._rated_powers[i] / 1e3 * 8760.0) * 100.0) if self._rated_powers[i] > 0 else 0.0,
                avg_effective_speed=0.0,
                dominant_wake_source=dominant_source,
                total_power_loss_by_source=loss_sources,
            )
            turbine_results.append(turb_result)

        return FarmResult(
            gross_aep=gross_aep,
            net_aep=net_aep,
            total_wake_loss=total_loss,
            wake_loss_pct=wake_loss_pct,
            capacity_factor=capacity_factor,
            total_installed_capacity=total_installed,
            turbine_results=turbine_results,
            sector_results=sector_results,
        )

    def evaluate_layout(
        self,
        positions: np.ndarray,
    ) -> float:
        """快速评估布局，仅返回净AEP（用于优化器）。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)

        Returns
        -------
        float
            净AEP (MWh/year)
        """
        positions = np.asarray(positions, dtype=np.float64)

        n_turb = len(self.turbines)
        net_aep = 0.0

        for s_idx in range(self.wind_resource.num_sectors):
            sector = self.wind_resource.sectors[s_idx]
            freq = sector.frequency
            wind_dir = sector.direction_center
            k = sector.weibull_k
            c = sector.weibull_c

            pdf = self.wind_resource.weibull_pdf(self._speed_centers, s_idx)
            prob = pdf * self.speed_step

            total_deficit = self._compute_wake_deficit_field(positions, wind_dir)

            effective_speeds = self._speed_centers[np.newaxis, :] * (1.0 - total_deficit[:, np.newaxis])

            for i in range(n_turb):
                power = np.interp(
                    effective_speeds[i],
                    self._power_curves[i][:, 0],
                    self._power_curves[i][:, 1],
                    left=0.0,
                    right=0.0,
                )
                net_aep += float(np.sum(power * prob * 8760.0 * freq))

        return net_aep / 1e3
