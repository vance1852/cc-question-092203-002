"""详细结果计算与返回语义的回归测试。

覆盖三个缺陷的修复：
1. 逐机明细中的平均有效风速不再恒为 0，而是按扇区频率和风速概率加权；
2. return_details=False 真正跳过明细计算，不再返回整套逐机/扇区对象；
3. 多重尾流下各来源损失按对叠加亏损的贡献比例分摊，
   分摊之和恒等于该风机的实际总损失（守恒、不重复计算）。

并验证空场景、单机、多上游源等边界情形，以及汇总值、逐机值、
扇区值与保存到结果文件的数据可以相互核对。
"""

import json
from unittest import mock

import numpy as np
import pytest

from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wind_resource import (
    WindResource,
    WindSector,
    create_default_wind_resource,
)
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.farm.aep import AEPCalculator


def make_single_sector_resource() -> WindResource:
    """单扇区风资源：风始终从 270° 吹来（沿 +x 方向），便于构造受控尾流。"""
    return WindResource(
        [
            WindSector(
                direction_center=270.0,
                direction_width=360.0,
                frequency=1.0,
                mean_speed=8.5,
                weibull_k=2.0,
                weibull_c=9.6,
            )
        ]
    )


def make_turbines(n: int):
    return [create_default_turbine("V126-3.45MW") for _ in range(n)]


def make_calculator(
    turbines,
    resource,
    superposition: str = "sum_of_squares",
) -> AEPCalculator:
    return AEPCalculator(
        turbines=turbines,
        wind_resource=resource,
        wake_model=JensenWake(wake_decay=0.07),
        wake_superposition=superposition,
        speed_step=0.5,
    )


# 三机沿风向排成一列，下游风机同时受两个上游源影响（多重尾流场景）
ROW_POSITIONS = np.array([[0.0, 0.0], [700.0, 0.0], [1400.0, 0.0]])


class TestAvgEffectiveSpeed:
    """平均有效风速：由扇区频率和风速概率正确加权。"""

    def test_single_turbine_equals_free_stream_mean(self):
        resource = make_single_sector_resource()
        calc = make_calculator(make_turbines(1), resource)
        result = calc.compute_farm_aep(np.array([[0.0, 0.0]]))

        tr = result.turbine_results[0]
        assert tr.avg_effective_speed > 0.0
        # 无尾流时，平均有效风速应等于自由来流的概率加权平均风速
        assert tr.avg_effective_speed == pytest.approx(
            resource.overall_mean_speed, rel=0.02
        )

    def test_wake_reduces_effective_speed_by_deficit(self):
        resource = make_single_sector_resource()
        calc = make_calculator(make_turbines(2), resource)
        positions = np.array([[0.0, 0.0], [800.0, 0.0]])
        result = calc.compute_farm_aep(positions)

        upstream, downstream = result.turbine_results
        # 亏损与风速无关时，下游平均有效风速 = (1 - 叠加亏损) × 上游平均有效风速
        deficit = calc._compute_wake_deficit_field(positions, 270.0)[1]
        assert deficit > 0.0
        assert downstream.avg_effective_speed == pytest.approx(
            (1.0 - deficit) * upstream.avg_effective_speed, rel=1e-9
        )
        assert downstream.avg_effective_speed < upstream.avg_effective_speed

    def test_multi_sector_weighting_within_free_stream_bounds(self):
        resource = create_default_wind_resource(
            num_sectors=12, dominant_direction=270.0, mean_speed=8.5
        )
        calc = make_calculator(make_turbines(3), resource)
        result = calc.compute_farm_aep(ROW_POSITIONS)

        free_mean = resource.overall_mean_speed
        for tr in result.turbine_results:
            assert 0.0 < tr.avg_effective_speed <= free_mean + 1e-6


class TestLossAttributionConservation:
    """来源归因：叠加尾流场景下守恒且不重复计算。"""

    @pytest.mark.parametrize("superposition", ["sum_of_squares", "linear"])
    def test_sources_sum_to_actual_loss_multiple_upstreams(self, superposition):
        resource = make_single_sector_resource()
        calc = make_calculator(make_turbines(3), resource, superposition)
        result = calc.compute_farm_aep(ROW_POSITIONS)

        # 末位风机同时受两个上游源影响
        last = result.turbine_results[2]
        assert set(last.total_power_loss_by_source) == {0, 1}

        for tr in result.turbine_results:
            assert sum(tr.total_power_loss_by_source.values()) == pytest.approx(
                tr.wake_loss, rel=1e-9, abs=1e-9
            )

    @pytest.mark.parametrize("superposition", ["sum_of_squares", "linear"])
    def test_attribution_shares_follow_deficit_contributions(self, superposition):
        resource = make_single_sector_resource()
        calc = make_calculator(make_turbines(3), resource, superposition)
        result = calc.compute_farm_aep(ROW_POSITIONS)

        deficit_matrix = calc._compute_deficit_matrix(ROW_POSITIONS, 270.0)
        if superposition == "sum_of_squares":
            contrib = deficit_matrix**2
        else:
            contrib = deficit_matrix

        last = result.turbine_results[2]
        shares = contrib[:, 2] / contrib[:, 2].sum()
        assert last.dominant_wake_source == int(np.argmax(contrib[:, 2]))
        for src, loss in last.total_power_loss_by_source.items():
            assert loss == pytest.approx(shares[src] * last.wake_loss, rel=1e-9)

    def test_no_wake_no_sources(self):
        resource = make_single_sector_resource()
        calc = make_calculator(make_turbines(2), resource)
        # 垂直于风向布置，互不在对方尾流中
        result = calc.compute_farm_aep(np.array([[0.0, 0.0], [0.0, 2000.0]]))

        for tr in result.turbine_results:
            assert tr.wake_loss == pytest.approx(0.0, abs=1e-9)
            assert tr.total_power_loss_by_source == {}
            assert tr.dominant_wake_source is None


class TestReturnDetailsSemantics:
    """return_details=False：不返回明细对象，且真正跳过明细计算。"""

    def test_details_disabled_returns_no_detail_objects(self):
        resource = make_single_sector_resource()
        calc = make_calculator(make_turbines(3), resource)

        detailed = calc.compute_farm_aep(ROW_POSITIONS, return_details=True)
        fast = calc.compute_farm_aep(ROW_POSITIONS, return_details=False)

        assert detailed.turbine_results and detailed.sector_results
        assert fast.turbine_results == []
        assert fast.sector_results == {}

        # 汇总指标不受明细开关影响
        assert fast.gross_aep == detailed.gross_aep
        assert fast.net_aep == detailed.net_aep
        assert fast.total_wake_loss == detailed.total_wake_loss
        assert fast.wake_loss_pct == detailed.wake_loss_pct
        assert fast.capacity_factor == detailed.capacity_factor
        assert fast.total_installed_capacity == detailed.total_installed_capacity

    def test_details_disabled_skips_attribution_computation(self):
        resource = make_single_sector_resource()
        calc = make_calculator(make_turbines(3), resource)

        with mock.patch.object(
            calc, "_attribute_loss_by_source", wraps=calc._attribute_loss_by_source
        ) as spy:
            calc.compute_farm_aep(ROW_POSITIONS, return_details=False)
            assert spy.call_count == 0

            calc.compute_farm_aep(ROW_POSITIONS, return_details=True)
            assert spy.call_count == resource.num_sectors

    def test_evaluate_layout_consistent_with_compute_farm_aep(self):
        resource = create_default_wind_resource(
            num_sectors=12, dominant_direction=270.0, mean_speed=8.5
        )
        calc = make_calculator(make_turbines(3), resource)

        detailed = calc.compute_farm_aep(ROW_POSITIONS)
        fast = calc.compute_farm_aep(ROW_POSITIONS, return_details=False)
        assert calc.evaluate_layout(ROW_POSITIONS) == fast.net_aep == detailed.net_aep


class TestEdgeCases:
    """空场景与单机场景。"""

    def test_empty_farm_returns_zeros(self):
        resource = make_single_sector_resource()
        calc = make_calculator([], resource)
        positions = np.zeros((0, 2))

        result = calc.compute_farm_aep(positions)
        assert result.gross_aep == 0.0
        assert result.net_aep == 0.0
        assert result.total_wake_loss == 0.0
        assert result.wake_loss_pct == 0.0
        assert result.capacity_factor == 0.0
        assert result.total_installed_capacity == 0.0
        assert result.turbine_results == []
        # 扇区明细仍然完整且可核对（全部为零）
        assert len(result.sector_results) == resource.num_sectors
        for sector in result.sector_results.values():
            assert sector["net_aep"] == 0.0
            assert sector["gross_aep"] == 0.0
            assert sector["wake_loss"] == 0.0

        fast = calc.compute_farm_aep(positions, return_details=False)
        assert fast.net_aep == 0.0
        assert calc.evaluate_layout(positions) == 0.0

    def test_single_turbine_has_no_wake(self):
        resource = make_single_sector_resource()
        calc = make_calculator(make_turbines(1), resource)
        result = calc.compute_farm_aep(np.array([[500.0, 500.0]]))

        tr = result.turbine_results[0]
        assert tr.net_aep == tr.gross_aep
        assert tr.wake_loss == 0.0
        assert tr.wake_loss_pct == 0.0
        assert tr.dominant_wake_source is None
        assert tr.total_power_loss_by_source == {}
        assert result.net_aep == tr.net_aep
        assert result.total_wake_loss == 0.0


class TestCrossCheckConsistency:
    """汇总值、逐机值、扇区值与结果文件必须能相互核对。"""

    POSITIONS = np.array(
        [[0, 0], [800, 0], [1600, 0], [2400, 0], [800, 900], [1600, 900]],
        dtype=float,
    )

    def make_result(self):
        resource = create_default_wind_resource(
            num_sectors=12, dominant_direction=270.0, mean_speed=8.5
        )
        calc = make_calculator(make_turbines(6), resource)
        return calc.compute_farm_aep(self.POSITIONS), resource

    def test_farm_equals_sum_of_turbines_and_sectors(self):
        result, _ = self.make_result()

        sum_net = sum(tr.net_aep for tr in result.turbine_results)
        sum_gross = sum(tr.gross_aep for tr in result.turbine_results)
        assert sum_net == pytest.approx(result.net_aep, rel=1e-9)
        assert sum_gross == pytest.approx(result.gross_aep, rel=1e-9)

        sector_net = sum(s["net_aep"] for s in result.sector_results.values())
        sector_gross = sum(s["gross_aep"] for s in result.sector_results.values())
        sector_loss = sum(s["wake_loss"] for s in result.sector_results.values())
        assert sector_net == pytest.approx(result.net_aep, rel=1e-9)
        assert sector_gross == pytest.approx(result.gross_aep, rel=1e-9)
        assert sector_loss == pytest.approx(result.total_wake_loss, rel=1e-9)

        for s in result.sector_results.values():
            assert s["gross_aep"] - s["net_aep"] == pytest.approx(
                s["wake_loss"], rel=1e-9
            )

    def test_turbine_loss_breakdown_is_explainable(self):
        result, resource = self.make_result()

        sum_loss = 0.0
        for tr in result.turbine_results:
            assert tr.gross_aep - tr.net_aep == pytest.approx(
                tr.wake_loss, rel=1e-9
            )
            # 各来源分摊之和恒等于该机实际总损失
            assert sum(tr.total_power_loss_by_source.values()) == pytest.approx(
                tr.wake_loss, rel=1e-9, abs=1e-9
            )
            if tr.total_power_loss_by_source:
                assert tr.dominant_wake_source == max(
                    tr.total_power_loss_by_source,
                    key=tr.total_power_loss_by_source.get,
                )
            sum_loss += tr.wake_loss

        assert sum_loss == pytest.approx(result.total_wake_loss, rel=1e-9)
        assert result.gross_aep - result.net_aep == pytest.approx(
            result.total_wake_loss, rel=1e-12
        )

        # 该布局存在尾流：至少一台风机的平均有效风速明显低于自由来流
        free_mean = resource.overall_mean_speed
        assert any(
            tr.avg_effective_speed < 0.9 * free_mean
            for tr in result.turbine_results
        )

    def test_saved_results_file_cross_check(self, tmp_path):
        from wind_farm_opt.config import (
            WindFarmConfig,
            OptimizationConfig,
            VisualizationConfig,
            EconomicConfig,
        )
        from wind_farm_opt.cli import WindFarmOptimizerCLI

        config = WindFarmConfig(
            n_turbines=4,
            optimization=OptimizationConfig(
                population_size=6, max_iterations=2, seed=42
            ),
            visualization=VisualizationConfig(
                save_dir=str(tmp_path), save_plots=False, plot_wake_heatmap=False
            ),
            economic=EconomicConfig(enable_analysis=False),
        )
        cli = WindFarmOptimizerCLI(config)
        cli.run_baseline()
        cli.save_results()

        with open(tmp_path / "results.json", encoding="utf-8") as f:
            saved = json.load(f)

        baseline = saved["baseline"]
        turbines = baseline["turbine_losses"]
        sectors = baseline["sectors"]

        # 文件中的汇总值与逐机值、扇区值相互核对
        assert sum(t["net_aep_mwh"] for t in turbines) == pytest.approx(
            baseline["net_aep_gwh"] * 1e3, rel=1e-9
        )
        assert sum(t["gross_aep_mwh"] for t in turbines) == pytest.approx(
            baseline["gross_aep_gwh"] * 1e3, rel=1e-9
        )
        assert sum(s["net_aep_mwh"] for s in sectors) == pytest.approx(
            baseline["net_aep_gwh"] * 1e3, rel=1e-9
        )
        assert sum(s["gross_aep_mwh"] for s in sectors) == pytest.approx(
            baseline["gross_aep_gwh"] * 1e3, rel=1e-9
        )

        # 文件中每台风机的净发电量构成可解释：
        # 理论 - 损失 = 净发电量；各来源分摊之和 = 该机总损失
        for t in turbines:
            assert t["gross_aep_mwh"] - t["net_aep_mwh"] == pytest.approx(
                t["wake_loss_mwh"], rel=1e-9
            )
            assert sum(t["loss_by_source_mwh"].values()) == pytest.approx(
                t["wake_loss_mwh"], rel=1e-9, abs=1e-9
            )
            assert t["avg_effective_speed"] > 0.0
