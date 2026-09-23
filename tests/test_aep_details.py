"""AEP 详细结果计算与返回语义的测试。

覆盖：
- 平均有效风速按扇区频率与威布尔概率正确加权；
- 多尾流叠加下来源归因守恒、不重复计算；
- return_details=False 时跳过明细分析且汇总值不变；
- 汇总值 / 逐机值 / 扇区值 / 序列化结果互相可核对；
- 空场景、单机、多个上游源等边界情形。
"""

import json

import numpy as np
import pytest

from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.core.wind_resource import WindResource, WindSector
from wind_farm_opt.farm.aep import AEPCalculator, HOURS_PER_YEAR


# --------------------------------------------------------------------------
# 测试夹具
# --------------------------------------------------------------------------


def make_resource(directions=(270.0,), frequencies=(1.0,), mean_speed=8.5, k=2.2):
    """构造单扇区或多扇区风资源。"""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    frequencies = frequencies / frequencies.sum()
    # c 使威布尔均值近似 mean_speed（k 固定，c = mean / Gamma(1+1/k)）
    from wind_farm_opt.core.wind_resource import _gamma_lanczos

    c = mean_speed / float(_gamma_lanczos(1.0 + 1.0 / k))
    sectors = [
        WindSector(
            direction_center=float(d),
            direction_width=360.0 / len(directions),
            frequency=float(f),
            mean_speed=mean_speed,
            weibull_k=k,
            weibull_c=c,
        )
        for d, f in zip(directions, frequencies)
    ]
    return WindResource(sectors)


def make_calculator(n_turbines, resource=None, superposition="sum_of_squares",
                    speed_step=0.5):
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turbines)]
    return AEPCalculator(
        turbines=turbines,
        wind_resource=resource or make_resource(),
        wake_model=JensenWake(wake_decay=0.07),
        wake_superposition=superposition,
        speed_step=speed_step,
    )


# 风向 270° 时风矢量沿 +x 方向，x 增大为下游
DOWNSTREAM_POSITIONS_3 = np.array([[0.0, 0.0], [700.0, 0.0], [1400.0, 0.0]])


# --------------------------------------------------------------------------
# 平均有效风速
# --------------------------------------------------------------------------


class TestAverageEffectiveSpeed:
    def test_single_turbine_matches_weibull_mean(self):
        """无尾流时平均有效风速应等于该扇区威布尔均值（概率加权）。"""
        resource = make_resource(mean_speed=8.5, k=2.2)
        calc = make_calculator(1, resource=resource, speed_step=0.25)
        result = calc.compute_farm_aep(np.array([[0.0, 0.0]]))

        tr = result.turbine_results[0]
        expected = resource.weibull_mean(0)
        assert tr.avg_effective_speed == pytest.approx(expected, abs=5e-2)
        assert tr.avg_effective_speed > 0.0

    def test_effective_speed_decreases_in_wake(self):
        calc = make_calculator(2)
        result = calc.compute_farm_aep(np.array([[0.0, 0.0], [700.0, 0.0]]))

        free, waked = result.turbine_results
        assert waked.avg_effective_speed < free.avg_effective_speed
        assert 0.0 < waked.avg_effective_speed < free.avg_effective_speed

    def test_weighted_across_sectors(self):
        """两个扇区不同均值时，平均有效风速应为按频率的加权平均。"""
        res = make_resource(directions=(90.0, 270.0), frequencies=(0.3, 0.7),
                            mean_speed=9.0, k=2.5)
        calc = make_calculator(1, resource=res, speed_step=0.25)
        # 风机位于与两扇区都不对齐的位置，单机无尾流
        result = calc.compute_farm_aep(np.array([[0.0, 0.0]]))
        # 两个扇区均值相同（均为 9 m/s 的威布尔分布）
        assert result.turbine_results[0].avg_effective_speed == pytest.approx(
            res.weibull_mean(0), abs=5e-2
        )

    def test_details_off_still_computes_speed(self):
        """关闭明细不应影响平均有效风速。"""
        calc = make_calculator(2)
        pos = np.array([[0.0, 0.0], [700.0, 0.0]])
        on = calc.compute_farm_aep(pos, return_details=True)
        off = calc.compute_farm_aep(pos, return_details=False)
        for a, b in zip(on.turbine_results, off.turbine_results):
            assert b.avg_effective_speed == pytest.approx(a.avg_effective_speed)


# --------------------------------------------------------------------------
# 来源归因守恒
# --------------------------------------------------------------------------


class TestSourceAttributionConservation:
    def test_sources_sum_to_turbine_loss(self):
        """每台风机各来源损失之和必须恰好等于其实际总尾流损失。"""
        calc = make_calculator(3)
        result = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3)

        for tr in result.turbine_results:
            attributed = sum(tr.total_power_loss_by_source.values())
            assert attributed == pytest.approx(tr.wake_loss, abs=1e-9), (
                f"风机 {tr.turbine_idx} 来源归因不守恒: "
                f"{attributed} != {tr.wake_loss}"
            )

    def test_two_upstream_sources(self):
        """一台下游风机同时受两台上游风机影响：归因守恒且来源齐全。"""
        # 两台上游风机横向错开 100 m，都在下游风机的尾流锥内
        pos = np.array([[0.0, -100.0], [0.0, 100.0], [700.0, 0.0]])
        calc = make_calculator(3)
        result = calc.compute_farm_aep(pos)

        downstream = result.turbine_results[2]
        assert downstream.wake_loss > 0.0
        assert set(downstream.total_power_loss_by_source) == {0, 1}
        assert sum(downstream.total_power_loss_by_source.values()) == pytest.approx(
            downstream.wake_loss, abs=1e-9
        )
        # 对称布置 → 两来源应近似均摊
        losses = list(downstream.total_power_loss_by_source.values())
        assert losses[0] == pytest.approx(losses[1], rel=1e-6)
        assert downstream.dominant_wake_source in (0, 1)

    def test_no_double_count_under_linear_superposition(self):
        """线性叠加（最容易重复计算）下来源之和仍不得超过实际损失。"""
        calc = make_calculator(3, superposition="linear")
        result = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3)

        total_attributed = 0.0
        for tr in result.turbine_results:
            attributed = sum(tr.total_power_loss_by_source.values())
            assert attributed <= tr.wake_loss + 1e-9
            assert attributed == pytest.approx(tr.wake_loss, abs=1e-9)
            total_attributed += attributed
        assert total_attributed == pytest.approx(result.total_wake_loss, abs=1e-9)

    def test_sector_allocations_sum_to_sector_loss(self):
        """扇区级归因同样要守恒。"""
        resource = make_resource(directions=(90.0, 270.0), frequencies=(0.4, 0.6))
        calc = make_calculator(3, resource=resource)
        result = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3)

        assert len(result.sector_results) == 2
        for s_idx, sector in result.sector_results.items():
            turbine_losses = sector["turbine_wake_loss"]
            for j, sources in enumerate(sector["loss_by_source"]):
                assert sum(sources.values()) == pytest.approx(
                    turbine_losses[j], abs=1e-9
                )
            assert sum(turbine_losses) == pytest.approx(sector["wake_loss"], abs=1e-9)

    def test_free_turbine_has_no_sources(self):
        """不在任何尾流中的风机损失与来源归因都应为零。"""
        # 风向沿 +x，两机沿 y 方向拉开 → 完全侧向，无尾流
        pos = np.array([[0.0, 0.0], [0.0, 2000.0]])
        calc = make_calculator(2)
        result = calc.compute_farm_aep(pos)

        for tr in result.turbine_results:
            assert tr.wake_loss == pytest.approx(0.0, abs=1e-9)
            assert tr.total_power_loss_by_source == {}
            assert tr.dominant_wake_source is None


# --------------------------------------------------------------------------
# 汇总 / 逐机 / 扇区一致性
# --------------------------------------------------------------------------


class TestResultReconciliation:
    def test_farm_equals_sum_of_turbines(self):
        calc = make_calculator(3)
        result = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3)

        assert sum(t.gross_aep for t in result.turbine_results) == pytest.approx(
            result.gross_aep
        )
        assert sum(t.net_aep for t in result.turbine_results) == pytest.approx(
            result.net_aep
        )
        assert sum(t.wake_loss for t in result.turbine_results) == pytest.approx(
            result.total_wake_loss
        )
        assert result.total_wake_loss == pytest.approx(
            result.gross_aep - result.net_aep
        )

    def test_sectors_sum_to_farm(self):
        resource = make_resource(directions=(0.0, 90.0, 180.0, 270.0),
                                 frequencies=(1.0, 2.0, 3.0, 4.0))
        calc = make_calculator(3, resource=resource)
        result = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3)

        assert sum(s["gross_aep"] for s in result.sector_results.values()) == pytest.approx(
            result.gross_aep
        )
        assert sum(s["net_aep"] for s in result.sector_results.values()) == pytest.approx(
            result.net_aep
        )
        assert sum(s["wake_loss"] for s in result.sector_results.values()) == pytest.approx(
            result.total_wake_loss
        )

        # 扇区 × 风机的矩阵两个方向求和都要对得上
        sector_turbine_net = np.array(
            [s["turbine_net_aep"] for s in result.sector_results.values()]
        )
        assert sector_turbine_net.sum(axis=0) == pytest.approx(
            [t.net_aep for t in result.turbine_results]
        )

        # 全场平均有效风速 = 各扇区（条件）平均有效风速按频率加权
        sector_speeds = np.array(
            [s["turbine_avg_effective_speed"] for s in result.sector_results.values()]
        )
        sector_freqs = np.array(
            [s["frequency"] for s in result.sector_results.values()]
        )
        weighted = sector_speeds.T @ sector_freqs / sector_freqs.sum()
        assert weighted == pytest.approx(
            [t.avg_effective_speed for t in result.turbine_results], abs=1e-9
        )

    def test_sector_frequencies_present(self):
        resource = make_resource(directions=(90.0, 270.0), frequencies=(0.3, 0.7))
        calc = make_calculator(3, resource=resource)
        result = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3)
        freqs = {s["direction"]: s["frequency"] for s in result.sector_results.values()}
        assert freqs[90.0] == pytest.approx(0.3)
        assert freqs[270.0] == pytest.approx(0.7)


# --------------------------------------------------------------------------
# return_details 语义与开销
# --------------------------------------------------------------------------


class TestReturnDetails:
    def test_details_off_omits_sector_objects(self):
        calc = make_calculator(3)
        result = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3, return_details=False)
        assert result.sector_results == {}
        for tr in result.turbine_results:
            assert tr.total_power_loss_by_source == {}
            assert tr.dominant_wake_source is None

    def test_details_off_totals_unchanged(self):
        calc = make_calculator(3)
        on = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3, return_details=True)
        off = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3, return_details=False)

        for attr in ("gross_aep", "net_aep", "total_wake_loss", "wake_loss_pct",
                     "capacity_factor", "total_installed_capacity"):
            assert getattr(off, attr) == pytest.approx(getattr(on, attr))
        for a, b in zip(on.turbine_results, off.turbine_results):
            assert a.gross_aep == pytest.approx(b.gross_aep)
            assert a.net_aep == pytest.approx(b.net_aep)
            assert a.avg_effective_speed == pytest.approx(b.avg_effective_speed)

    def test_details_off_reduces_interpolation_work(self):
        """关闭明细必须真正减少功率曲线插值这一主要分析开销。"""
        calc = make_calculator(4)
        pos = np.array(
            [[0.0, 0.0], [700.0, 0.0], [0.0, 700.0], [700.0, 700.0]]
        )

        counts = {"n": 0}
        original = calc._power_at

        def counting_power(idx, speeds):
            counts["n"] += np.asarray(speeds).size
            return original(idx, speeds)

        calc._power_at = counting_power
        calc.compute_farm_aep(pos, return_details=True)
        on_count = counts["n"]
        counts["n"] = 0
        calc.compute_farm_aep(pos, return_details=False)
        off_count = counts["n"]

        assert off_count < on_count

    def test_evaluate_layout_matches_net_aep(self):
        calc = make_calculator(3)
        result = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3, return_details=False)
        fast = calc.evaluate_layout(DOWNSTREAM_POSITIONS_3)
        assert fast == pytest.approx(result.net_aep, rel=1e-12)


# --------------------------------------------------------------------------
# 边界场景
# --------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_farm(self):
        calc = make_calculator(0)
        result = calc.compute_farm_aep(np.zeros((0, 2)))

        assert result.gross_aep == 0.0
        assert result.net_aep == 0.0
        assert result.total_wake_loss == 0.0
        assert result.wake_loss_pct == 0.0
        assert result.turbine_results == []
        # 明细开关都不应导致空场景出错
        assert calc.compute_farm_aep(np.zeros((0, 2)),
                                     return_details=False).sector_results == {}

    def test_single_turbine(self):
        calc = make_calculator(1)
        result = calc.compute_farm_aep(np.array([[123.0, 456.0]]))

        tr = result.turbine_results[0]
        assert tr.gross_aep == pytest.approx(result.gross_aep)
        assert tr.net_aep == pytest.approx(result.net_aep)
        assert tr.wake_loss == pytest.approx(0.0, abs=1e-9)
        assert tr.total_power_loss_by_source == {}
        # 容量系数与净发电量自洽
        expected_cf = tr.net_aep / (3.45 * HOURS_PER_YEAR) * 100.0
        assert tr.capacity_factor == pytest.approx(expected_cf)

    def test_many_upstream_chain(self):
        """5 机串联：每台下游风机的归因都守恒，主导来源为最近的上游。"""
        calc = make_calculator(5)
        pos = np.array([[700.0 * i, 0.0] for i in range(5)])
        result = calc.compute_farm_aep(pos)

        for i, tr in enumerate(result.turbine_results):
            assert sum(tr.total_power_loss_by_source.values()) == pytest.approx(
                tr.wake_loss, abs=1e-9
            )
            assert all(0 <= src < i for src in tr.total_power_loss_by_source)
            if i > 0:
                assert tr.dominant_wake_source == i - 1

    def test_invalid_positions_shape_raises(self):
        calc = make_calculator(2)
        with pytest.raises(ValueError):
            calc.compute_farm_aep(np.array([[0.0, 0.0]]))


# --------------------------------------------------------------------------
# 序列化与保存数据
# --------------------------------------------------------------------------


class TestSerialization:
    def test_farm_result_to_dict_roundtrip(self):
        calc = make_calculator(3)
        result = calc.compute_farm_aep(DOWNSTREAM_POSITIONS_3)
        data = result.to_dict()

        # 必须可 JSON 序列化
        encoded = json.dumps(data)
        decoded = json.loads(encoded)

        assert decoded["gross_aep_mwh"] == pytest.approx(result.gross_aep)
        assert decoded["net_aep_mwh"] == pytest.approx(result.net_aep)
        assert len(decoded["turbines"]) == 3

        # 文件中的汇总值、逐机值、扇区值必须互相核对
        assert sum(t["net_aep_mwh"] for t in decoded["turbines"]) == pytest.approx(
            decoded["net_aep_mwh"]
        )
        sector_net = sum(s["net_aep_mwh"] for s in decoded["sectors"].values())
        assert sector_net == pytest.approx(decoded["net_aep_mwh"])

        # 来源归因在序列化后仍守恒（键转为字符串）
        for t in decoded["turbines"]:
            assert sum(t["wake_loss_by_source_mwh"].values()) == pytest.approx(
                t["wake_loss_mwh"], abs=1e-9
            )

        # 有效风速被写入结果文件
        assert decoded["turbines"][1]["avg_effective_speed_mps"] > 0.0

    def test_cli_result_serializer_keeps_legacy_fields(self):
        from wind_farm_opt.cli import WindFarmOptimizerCLI

        calc = make_calculator(2)
        result = calc.compute_farm_aep(np.array([[0.0, 0.0], [700.0, 0.0]]))
        # _result_to_json 不依赖实例状态
        data = WindFarmOptimizerCLI._result_to_json(None, result)

        assert data["gross_aep_gwh"] == pytest.approx(result.gross_aep / 1e3)
        assert data["net_aep_gwh"] == pytest.approx(result.net_aep / 1e3)
        assert len(data["turbine_losses"]) == 2
        assert data["turbine_losses"][1]["dominant_source"] == 0
        # 新层级数据同时存在
        assert len(data["turbines"]) == 2
        assert "sectors" in data
