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
        平均有效风速 (m/s)，按扇区频率和风速概率加权
    dominant_wake_source : Optional[int]
        主要尾流来源风机索引
    total_power_loss_by_source : dict[int, float]
        各来源风机分摊到的损失 (MWh/year)。
        按各来源对叠加后总速度亏损的相对贡献分摊，
        对所有来源求和恒等于本机的 wake_loss（守恒、不重复计算）。
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
        每台风机的详细结果；当 compute_farm_aep(return_details=False)
        时为空列表
    sector_results : dict[int, dict]
        每个扇区的详细结果；当 compute_farm_aep(return_details=False)
        时为空字典
    """

    gross_aep: float
    net_aep: float
    total_wake_loss: float
    wake_loss_pct: float
    capacity_factor: float
    total_installed_capacity: float
    turbine_results: list[TurbineResult]
    sector_results: dict[int, dict]


@dataclass
class _SectorResult:
    """单个风向扇区的中间计算结果（仅供内部使用）。

    Parameters
    ----------
    net_aep : np.ndarray
        每台风机的扇区净发电量 (N_turb,)，单位 kWh
    gross_aep : np.ndarray
        每台风机的扇区理论发电量 (N_turb,)，单位 kWh
    loss_by_source : Optional[np.ndarray]
        扇区内的损失归因矩阵 (N_turb, N_turb)，元素 [j, i] 为
        上游风机 i 分摊给下游风机 j 的损失 (kWh)；
        return_details=False 时为 None
    eff_speed_weighted : Optional[np.ndarray]
        有效风速加权的分子部分 (N_turb,)，即
        freq * Σ_bins prob * v_effective；return_details=False 时为 None
    prob_weight : float
        有效风速加权的分母部分，即 freq * Σ_bins prob
    """

    net_aep: np.ndarray
    gross_aep: np.ndarray
    loss_by_source: Optional[np.ndarray]
    eff_speed_weighted: Optional[np.ndarray]
    prob_weight: float


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

    def _compute_deficit_matrix(
        self,
        positions: np.ndarray,
        wind_direction: float,
    ) -> np.ndarray:
        """计算给定风向下两两风机之间的速度亏损矩阵。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)
        wind_direction : float
            风向 (度)

        Returns
        -------
        np.ndarray
            速度亏损矩阵 (N_turb, N_turb)，元素 [i, j] 为上游风机 i
            在下游风机 j 处产生的速度亏损（未叠加）
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

        return deficit_matrix

    def _compute_wake_deficit_field(
        self,
        positions: np.ndarray,
        wind_direction: float,
    ) -> np.ndarray:
        """计算给定风向下每台风机叠加后的总速度亏损。

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
        deficit_matrix = self._compute_deficit_matrix(positions, wind_direction)
        return superpose_wakes(deficit_matrix, method=self.wake_superposition)

    def _attribute_loss_by_source(
        self,
        deficit_matrix: np.ndarray,
        total_loss_per_turbine: np.ndarray,
    ) -> np.ndarray:
        """将每台下游风机的实际尾流损失守恒地分摊给各上游来源。

        分摊权重取各来源对叠加后总速度亏损的相对贡献：
        平方和叠加正比于 deficit²，线性叠加正比于 deficit。
        因此对每台下游风机 j 恒有
        ``sum_i loss[j, i] == total_loss_per_turbine[j]``，
        在多重尾流重叠时既守恒也不会重复计算。

        Parameters
        ----------
        deficit_matrix : np.ndarray
            速度亏损矩阵 (N, N)，元素 [i, j] 为上游风机 i 在
            下游风机 j 处产生的速度亏损（未叠加）
        total_loss_per_turbine : np.ndarray
            每台风机的实际总损失 (N,)，单位 kWh

        Returns
        -------
        np.ndarray
            损失归因矩阵 (N, N)，元素 [j, i] 表示风机 i 分摊给
            风机 j 的损失 (kWh)
        """
        if self.wake_superposition == "sum_of_squares":
            contribution = deficit_matrix ** 2
        elif self.wake_superposition == "linear":
            contribution = deficit_matrix
        else:
            raise ValueError(f"未知的尾流叠加方法: {self.wake_superposition}")

        denom = contribution.sum(axis=0)
        weights = np.divide(
            contribution,
            denom[np.newaxis, :],
            out=np.zeros_like(contribution),
            where=denom[np.newaxis, :] > 0.0,
        )

        # weights 为 [上游, 下游]，转置为 [下游, 上游] 后乘以下游风机的实际总损失
        return weights.T * total_loss_per_turbine[:, np.newaxis]

    def _compute_sector_aep(
        self,
        positions: np.ndarray,
        sector_idx: int,
        return_details: bool = True,
    ) -> _SectorResult:
        """计算单个风向扇区的发电量。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)
        sector_idx : int
            扇区索引
        return_details : bool
            是否计算损失归因矩阵和有效风速加权等明细；
            为 False 时跳过这些额外计算

        Returns
        -------
        _SectorResult
            扇区计算结果
        """
        sector = self.wind_resource.sectors[sector_idx]
        freq = sector.frequency
        wind_dir = sector.direction_center

        pdf = self.wind_resource.weibull_pdf(self._speed_centers, sector_idx)
        prob = pdf * self.speed_step

        deficit_matrix = self._compute_deficit_matrix(positions, wind_dir)
        total_deficit = superpose_wakes(deficit_matrix, method=self.wake_superposition)

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

        loss_by_source = None
        eff_speed_weighted = None
        prob_weight = 0.0
        if return_details:
            total_loss_sector = gross_aep_sector - net_aep_sector
            loss_by_source = self._attribute_loss_by_source(
                deficit_matrix,
                total_loss_sector,
            )
            eff_speed_weighted = freq * np.sum(
                effective_speeds * prob[np.newaxis, :],
                axis=1,
            )
            prob_weight = freq * float(np.sum(prob))

        return _SectorResult(
            net_aep=net_aep_sector,
            gross_aep=gross_aep_sector,
            loss_by_source=loss_by_source,
            eff_speed_weighted=eff_speed_weighted,
            prob_weight=prob_weight,
        )

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
            是否返回详细结果。为 False 时跳过逐来源损失归因、
            逐机结果和扇区结果的构建（turbine_results 为空列表、
            sector_results 为空字典），仅计算全场汇总指标，
            以减少不必要的分析开销

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
        loss_by_source_total = np.zeros((n_turb, n_turb), dtype=np.float64) if return_details else None
        eff_speed_num = np.zeros(n_turb, dtype=np.float64) if return_details else None
        eff_speed_den = 0.0

        sector_results = {}

        for s_idx in range(self.wind_resource.num_sectors):
            sector_res = self._compute_sector_aep(
                positions,
                s_idx,
                return_details=return_details,
            )
            net_aep_by_turbine += sector_res.net_aep
            gross_aep_by_turbine += sector_res.gross_aep

            if return_details:
                loss_by_source_total += sector_res.loss_by_source
                eff_speed_num += sector_res.eff_speed_weighted
                eff_speed_den += sector_res.prob_weight

                sector_gross = float(np.sum(sector_res.gross_aep)) / 1e3
                sector_net = float(np.sum(sector_res.net_aep)) / 1e3
                sector_results[s_idx] = {
                    "direction": self.wind_resource.sectors[s_idx].direction_center,
                    "frequency": self.wind_resource.sectors[s_idx].frequency,
                    "net_aep": sector_net,
                    "gross_aep": sector_gross,
                    "wake_loss": sector_gross - sector_net,
                }

        gross_aep = float(np.sum(gross_aep_by_turbine)) / 1e3
        net_aep = float(np.sum(net_aep_by_turbine)) / 1e3
        total_loss = gross_aep - net_aep
        wake_loss_pct = (total_loss / gross_aep * 100.0) if gross_aep > 0 else 0.0

        total_installed = float(np.sum(self._rated_powers)) / 1e3
        capacity_factor = (net_aep / (total_installed * 8760.0) * 100.0) if total_installed > 0 else 0.0

        turbine_results = []
        if return_details:
            for i in range(n_turb):
                gross = gross_aep_by_turbine[i] / 1e3
                net = net_aep_by_turbine[i] / 1e3
                loss = gross - net
                loss_pct = (loss / gross * 100.0) if gross > 0 else 0.0

                avg_effective_speed = (
                    float(eff_speed_num[i] / eff_speed_den) if eff_speed_den > 0.0 else 0.0
                )

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
                    avg_effective_speed=avg_effective_speed,
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

        与 compute_farm_aep(positions, return_details=False) 的净AEP
        完全一致，跳过全部明细计算。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)

        Returns
        -------
        float
            净AEP (MWh/year)
        """
        return self.compute_farm_aep(positions, return_details=False).net_aep
