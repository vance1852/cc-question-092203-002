"""风电场年发电量(AEP)计算模块。

高效的向量化尾流计算，支持多风向扇区和威布尔分布积分。

结果一致性约定
--------------
无论是否请求明细（``return_details``），下列恒等式始终成立：

- ``farm.gross_aep == sum(t.gross_aep for t in turbine_results)``
- ``farm.net_aep   == sum(t.net_aep   for t in turbine_results)``
- ``farm.total_wake_loss == farm.gross_aep - farm.net_aep``
- 每个扇区的净/理论发电量等于该扇区内各风机对应值之和；
- 所有扇区之和等于全场值。

开启明细时，每台风机的来源归因满足**守恒**：

    sum(total_power_loss_by_source.values()) == wake_loss

做法是：先按各上游风机*单独*作用时造成的功率损失确定权重，
再把该风机在叠加尾流下的*实际*总损失按权重分摊给各来源，
从而避免多尾流重叠区域被重复计算。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.turbine import Turbine
from ..core.wind_resource import WindResource
from ..core.wake import WakeModel, superpose_wakes

HOURS_PER_YEAR = 8760.0
# 小于该能量（kWh）的来源归因视为数值噪声，不计入明细
_LOSS_EPS_KWH = 1e-9


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
        平均有效风速 (m/s)，按扇区频率与威布尔风速概率加权
    dominant_wake_source : Optional[int]
        主要尾流来源风机索引
    total_power_loss_by_source : dict[int, float]
        各上游风机分摊到的尾流损失 (MWh/year)，
        各来源之和恒等于 ``wake_loss``（仅在 return_details=True 时填充）
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

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典（能量单位均为 MWh/年）。"""
        return {
            "idx": self.turbine_idx,
            "name": self.name,
            "gross_aep_mwh": float(self.gross_aep),
            "net_aep_mwh": float(self.net_aep),
            "wake_loss_mwh": float(self.wake_loss),
            "wake_loss_pct": float(self.wake_loss_pct),
            "capacity_factor_pct": float(self.capacity_factor),
            "avg_effective_speed_mps": float(self.avg_effective_speed),
            "dominant_wake_source": self.dominant_wake_source,
            # JSON 键只能是字符串，解析方需自行转回 int
            "wake_loss_by_source_mwh": {
                str(k): float(v) for k, v in self.total_power_loss_by_source.items()
            },
        }


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
        每台风机的结果；逐机发电量与平均有效风速始终计算，
        来源归因仅在 return_details=True 时填充
    sector_results : dict[int, dict]
        每个扇区的详细结果；return_details=False 时为空字典
    """

    gross_aep: float
    net_aep: float
    total_wake_loss: float
    wake_loss_pct: float
    capacity_factor: float
    total_installed_capacity: float
    turbine_results: list[TurbineResult]
    sector_results: dict[int, dict]

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典。

        汇总、逐机、扇区三层数据一并导出，且数值由同一份计算结果取得，
        可直接互相核对（扇区净/理论发电量之和等于全场值）。
        """
        return {
            "gross_aep_mwh": float(self.gross_aep),
            "net_aep_mwh": float(self.net_aep),
            "wake_loss_mwh": float(self.total_wake_loss),
            "wake_loss_pct": float(self.wake_loss_pct),
            "capacity_factor_pct": float(self.capacity_factor),
            "installed_capacity_mw": float(self.total_installed_capacity),
            "n_turbines": len(self.turbine_results),
            "turbines": [tr.to_dict() for tr in self.turbine_results],
            "sectors": {
                str(s_idx): {
                    "direction_deg": float(s["direction"]),
                    "frequency": float(s["frequency"]),
                    "gross_aep_mwh": float(s["gross_aep"]),
                    "net_aep_mwh": float(s["net_aep"]),
                    "wake_loss_mwh": float(s["wake_loss"]),
                    "wake_loss_pct": float(s["wake_loss_pct"]),
                    "turbine_gross_aep_mwh": [float(v) for v in s["turbine_gross_aep"]],
                    "turbine_net_aep_mwh": [float(v) for v in s["turbine_net_aep"]],
                    "turbine_wake_loss_mwh": [float(v) for v in s["turbine_wake_loss"]],
                    "turbine_avg_effective_speed_mps": [
                        float(v) for v in s["turbine_avg_effective_speed"]
                    ],
                    "wake_loss_by_source_mwh": [
                        {str(k): float(v) for k, v in sources.items()}
                        for sources in s["loss_by_source"]
                    ],
                }
                for s_idx, s in sorted(self.sector_results.items())
            },
        }


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

    def _power_at(self, turbine_idx: int, speeds: np.ndarray) -> np.ndarray:
        """按指定风机的功率曲线插值功率（集中封装便于复用与测试）。"""
        curve = self._power_curves[turbine_idx]
        return np.interp(speeds, curve[:, 0], curve[:, 1], left=0.0, right=0.0)

    def _compute_deficit_matrix(
        self,
        positions: np.ndarray,
        wind_direction: float,
    ) -> np.ndarray:
        """计算给定风向下各风机单独产生的速度亏损矩阵。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)
        wind_direction : float
            风向 (度)

        Returns
        -------
        np.ndarray
            亏损数组 (N_turb, N_turb)，元素 ``[i, j]`` 为上游风机 i
            在下游风机 j 处*单独*产生的速度亏损 (1 - u/U0)。
        """
        n = positions.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)

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

    def _compute_sector_contribution(
        self,
        positions: np.ndarray,
        sector_idx: int,
        include_details: bool,
    ) -> dict:
        """计算单个风向扇区的发电量及（可选的）来源归因。

        Returns
        -------
        dict
            键包括：

            - ``gross_aep`` / ``net_aep``：逐风机发电量 (kWh)，形状 (N,)
            - ``effective_speed_sum``：逐风机按 freq*prob 加权的有效风速之和
            - ``probability_mass``：该扇区 freq*prob 的总权重
            - ``allocations``：(N, N) 损失分摊矩阵 (kWh)，``[j, i]`` 为
              上游 i 分摊给下游 j 的损失；仅 include_details 时存在
            - ``detail``：可直接放入 ``sector_results`` 的字典
        """
        sector = self.wind_resource.sectors[sector_idx]
        freq = sector.frequency
        wind_dir = sector.direction_center

        pdf = self.wind_resource.weibull_pdf(self._speed_centers, sector_idx)
        prob = pdf * self.speed_step
        prob_sum = float(np.sum(prob))
        # 能量权重：扇区内各风速一年持续的小时数
        energy_weight = freq * HOURS_PER_YEAR * prob
        # 风速权重：仅做概率加权（不含年小时数）
        speed_weight = freq * prob

        deficit_matrix = self._compute_deficit_matrix(positions, wind_dir)
        # 矩阵形状 (上游, 下游)，沿第 0 轴叠加各上游风机的亏损
        total_deficit = superpose_wakes(
            deficit_matrix,
            method=self.wake_superposition,
        )

        n_turb = len(self.turbines)
        n_speed = len(self._speed_centers)

        effective_speeds = self._speed_centers[np.newaxis, :] * (
            1.0 - total_deficit[:, np.newaxis]
        )

        net_power = np.zeros((n_turb, n_speed), dtype=np.float64)
        for i in range(n_turb):
            net_power[i] = self._power_at(i, effective_speeds[i])

        gross_power = self._power_lookup

        gross_aep_sector = np.sum(gross_power * energy_weight, axis=1)
        net_aep_sector = np.sum(net_power * energy_weight, axis=1)
        effective_speed_sum = np.sum(effective_speeds * speed_weight, axis=1)

        result: dict = {
            "gross_aep": gross_aep_sector,
            "net_aep": net_aep_sector,
            "effective_speed_sum": effective_speed_sum,
            "probability_mass": freq * prob_sum,
        }

        if not include_details:
            return result

        # ---- 来源归因（守恒分摊） -------------------------------------
        # 1) 各上游风机单独作用时，下游风机的发电量（矩阵按 [下游, 上游] 索引）
        equivalent_gross = np.zeros((n_turb, n_turb), dtype=np.float64)
        for j in range(n_turb):
            # deficit_matrix 按 [上游, 下游] 索引，取下游 j 的那一列
            single_wake_speeds = self._speed_centers[np.newaxis, :] * (
                1.0 - deficit_matrix[:, j][:, np.newaxis]
            )
            powered = self._power_at(j, single_wake_speeds.ravel()).reshape(
                n_turb, n_speed
            )
            equivalent_gross[j] = np.sum(powered * energy_weight, axis=1)

        # 2) 单独作用权重：功率曲线单调不减，故等价发电量 <= 理论发电量
        standalone_loss = np.maximum(
            0.0, gross_aep_sector[:, np.newaxis] - equivalent_gross
        )
        np.fill_diagonal(standalone_loss, 0.0)
        weight_sum = standalone_loss.sum(axis=1)

        # 3) 实际总损失按权重分摊；重叠区域不重复、总损失不遗漏。
        #    兜底：若所有单独尾流都太弱（例如均未把风机推离额定平台），
        #    而叠加后的总亏损却造成了损失，则按各上游速度亏损的
        #    能量暴露程度（亏损 × 理论发电量）分配权重。
        actual_loss = np.maximum(0.0, gross_aep_sector - net_aep_sector)
        need_fallback = (weight_sum <= 0.0) & (actual_loss > 0.0)
        if np.any(need_fallback):
            fallback_weight = deficit_matrix.T * np.maximum(gross_aep_sector, 0.0)[:, np.newaxis]
            np.fill_diagonal(fallback_weight, 0.0)
            fallback_sum = fallback_weight.sum(axis=1)
            use_fallback = need_fallback & (fallback_sum > 0.0)
            standalone_loss = np.where(
                use_fallback[:, np.newaxis], fallback_weight, standalone_loss
            )
            weight_sum = np.where(
                use_fallback, fallback_sum, weight_sum
            )
            # 理论上不该再出现零权重却有损失的情况；若仍有则均匀分摊
            still_undefined = need_fallback & ~use_fallback
            if np.any(still_undefined):
                uniform = np.zeros_like(standalone_loss)
                uniform[still_undefined] = 1.0
                np.fill_diagonal(uniform, 0.0)
                uniform_sum = uniform.sum(axis=1)
                standalone_loss = np.where(
                    still_undefined[:, np.newaxis], uniform, standalone_loss
                )
                weight_sum = np.where(still_undefined, uniform_sum, weight_sum)

        safe_sum = np.where(weight_sum > 0.0, weight_sum, 1.0)
        share = np.where(
            weight_sum[:, np.newaxis] > 0.0,
            standalone_loss / safe_sum[:, np.newaxis],
            0.0,
        )
        allocations = share * actual_loss[:, np.newaxis]
        result["allocations"] = allocations

        sector_gross = float(np.sum(gross_aep_sector)) / 1e3
        sector_net = float(np.sum(net_aep_sector)) / 1e3
        sector_loss = sector_gross - sector_net

        sector_speeds = (
            np.sum(effective_speeds * prob, axis=1) / prob_sum
            if prob_sum > 0.0
            else np.zeros(n_turb)
        )

        loss_by_source: list[dict[int, float]] = []
        for j in range(n_turb):
            sources = {
                i: float(allocations[j, i]) / 1e3
                for i in range(n_turb)
                if i != j and allocations[j, i] > _LOSS_EPS_KWH
            }
            loss_by_source.append(sources)

        result["detail"] = {
            "direction": wind_dir,
            "frequency": float(freq),
            "gross_aep": sector_gross,
            "net_aep": sector_net,
            "wake_loss": sector_loss,
            "wake_loss_pct": (sector_loss / sector_gross * 100.0)
            if sector_gross > 0.0
            else 0.0,
            "turbine_gross_aep": (gross_aep_sector / 1e3).tolist(),
            "turbine_net_aep": (net_aep_sector / 1e3).tolist(),
            "turbine_wake_loss": (actual_loss / 1e3).tolist(),
            "turbine_avg_effective_speed": sector_speeds.tolist(),
            "loss_by_source": loss_by_source,
        }

        return result

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
            是否计算并返回逐扇区明细与逐来源尾流损失归因。
            关闭时跳过 O(N_turb^2) 的来源归因插值与扇区对象构造，
            汇总值与逐机发电量、平均有效风速不受影响。

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
        effective_speed_sum = np.zeros(n_turb, dtype=np.float64)
        probability_mass = 0.0
        allocations_total = (
            np.zeros((n_turb, n_turb), dtype=np.float64) if return_details else None
        )

        sector_results: dict[int, dict] = {}

        for s_idx in range(self.wind_resource.num_sectors):
            contribution = self._compute_sector_contribution(
                positions, s_idx, include_details=return_details
            )
            gross_aep_by_turbine += contribution["gross_aep"]
            net_aep_by_turbine += contribution["net_aep"]
            effective_speed_sum += contribution["effective_speed_sum"]
            probability_mass += contribution["probability_mass"]

            if return_details:
                allocations_total += contribution["allocations"]
                sector_results[s_idx] = contribution["detail"]

        gross_aep = float(np.sum(gross_aep_by_turbine)) / 1e3
        net_aep = float(np.sum(net_aep_by_turbine)) / 1e3
        total_loss = gross_aep - net_aep
        wake_loss_pct = (total_loss / gross_aep * 100.0) if gross_aep > 0.0 else 0.0

        total_installed = float(np.sum(self._rated_powers)) / 1e3
        capacity_factor = (
            (net_aep / (total_installed * HOURS_PER_YEAR) * 100.0)
            if total_installed > 0.0
            else 0.0
        )

        avg_effective_speeds = (
            effective_speed_sum / probability_mass
            if probability_mass > 0.0
            else np.zeros(n_turb)
        )

        turbine_results = []
        for i in range(n_turb):
            gross = float(gross_aep_by_turbine[i]) / 1e3
            net = float(net_aep_by_turbine[i]) / 1e3
            loss = gross - net
            loss_pct = (loss / gross * 100.0) if gross > 0.0 else 0.0

            loss_sources: dict[int, float] = {}
            if return_details:
                loss_sources = {
                    j: float(allocations_total[i, j]) / 1e3
                    for j in range(n_turb)
                    if j != i and allocations_total[i, j] > _LOSS_EPS_KWH
                }

            dominant_source = None
            if loss_sources:
                dominant_source = max(loss_sources, key=loss_sources.get)

            rated_mw = self._rated_powers[i] / 1e3
            turb_result = TurbineResult(
                turbine_idx=i,
                name=self._turbine_names[i],
                gross_aep=gross,
                net_aep=net,
                wake_loss=loss,
                wake_loss_pct=float(loss_pct),
                capacity_factor=(
                    net / (rated_mw * HOURS_PER_YEAR) * 100.0 if rated_mw > 0.0 else 0.0
                ),
                avg_effective_speed=float(avg_effective_speeds[i]),
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

        不构造任何扇区/逐机/来源明细对象。

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

            pdf = self.wind_resource.weibull_pdf(self._speed_centers, s_idx)
            prob = pdf * self.speed_step

            deficit_matrix = self._compute_deficit_matrix(positions, wind_dir)
            total_deficit = superpose_wakes(
                np.moveaxis(deficit_matrix, 1, 0),
                method=self.wake_superposition,
            )

            effective_speeds = self._speed_centers[np.newaxis, :] * (
                1.0 - total_deficit[:, np.newaxis]
            )

            for i in range(n_turb):
                power = self._power_at(i, effective_speeds[i])
                net_aep += float(np.sum(power * prob * HOURS_PER_YEAR * freq))

        return net_aep / 1e3
