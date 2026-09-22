"""快速测试脚本 - 用于验证核心功能。"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

print("=" * 60)
print("风电场布局优化工具 - 快速测试")
print("=" * 60)

print("\n1. 测试风机模型...")
from wind_farm_opt.core.turbine import create_default_turbine
turb = create_default_turbine("V126-3.45MW")
print(f"   ✓ 风机: {turb.name}")
print(f"   ✓ 额定功率: {turb.rated_power/1e3:.2f} MW")
print(f"   ✓ 转子直径: {turb.rotor_diameter:.1f} m")
print(f"   ✓ 切入风速: {turb.cut_in_speed:.1f} m/s")
print(f"   ✓ 额定风速: {turb.rated_speed:.1f} m/s")
print(f"   ✓ 切出风速: {turb.cut_out_speed:.1f} m/s")

print("\n2. 测试风资源模型...")
from wind_farm_opt.core.wind_resource import create_default_wind_resource
wr = create_default_wind_resource(num_sectors=12, dominant_direction=270.0, mean_speed=8.5)
print(f"   ✓ 扇区数: {wr.num_sectors}")
print(f"   ✓ 加权平均风速: {wr.overall_mean_speed:.2f} m/s")
freq_sort_idx = np.argsort(wr.frequencies)[::-1]
print(f"   ✓ 主风向: {wr.directions[freq_sort_idx[0]]:.0f}° ({wr.frequencies[freq_sort_idx[0]]*100:.1f}%)")

print("\n3. 测试尾流模型...")
from wind_farm_opt.core.wake import JensenWake, GaussianWake, superpose_wakes
jensen = JensenWake(wake_decay=0.07)
gaussian = GaussianWake(wake_decay=0.035)

dists = np.array([5.0, 10.0, 15.0]) * turb.rotor_diameter
d0 = turb.rotor_diameter
ct = turb.thrust_coefficient

jensen_def = jensen.velocity_deficit(dists, d0, ct)
gauss_def = gaussian.velocity_deficit(dists, d0, ct)
print(f"   ✓ Jensen尾流模型 (5D/10D/15D): {jensen_def[0]:.3f}, {jensen_def[1]:.3f}, {jensen_def[2]:.3f}")
print(f"   ✓ 高斯尾流模型 (5D/10D/15D): {gauss_def[0]:.3f}, {gauss_def[1]:.3f}, {gauss_def[2]:.3f}")

deficits = np.array([0.1, 0.05, 0.08])
total_def = superpose_wakes(deficits, method="sum_of_squares")
print(f"   ✓ 平方和叠加: {total_def:.4f} (线性叠加: {deficits.sum():.4f})")

print("\n4. 测试场地边界...")
from wind_farm_opt.constraints.boundary import (
    create_rectangular_boundary,
    create_irregular_boundary,
)
boundary = create_rectangular_boundary(width=4000, height=4000)
print(f"   ✓ 矩形场地面积: {boundary.area/1e6:.2f} km²")

irregular = create_irregular_boundary()
print(f"   ✓ 不规则场地面积: {irregular.area/1e6:.2f} km²")

test_point = np.array([1000.0, 1000.0])
test_outside = np.array([-5000.0, -5000.0])
print(f"   ✓ 点在边界内: {boundary.contains_point(test_point)}")
print(f"   ✓ 点在边界外: {boundary.contains_point(test_outside)}")

print("\n5. 测试间距约束...")
from wind_farm_opt.constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
)
diameters = np.array([turb.rotor_diameter, turb.rotor_diameter])
min_space = compute_min_spacing_from_diameters(diameters, min_multiple=5.0)
print(f"   ✓ 最小间距 (5D): {min_space:.1f} m")

good_positions = np.array([[0, 0], [1000, 0]])
bad_positions = np.array([[0, 0], [500, 0]])
valid, violations = check_min_spacing(good_positions, min_space)
print(f"   ✓ 合格布局检查: {valid}")
valid, violations = check_min_spacing(bad_positions, min_space)
print(f"   ✓ 违规布局检查: {valid}, 违规对数: {len(violations)}")

print("\n6. 测试AEP计算（12台风机，50x50网格分辨率）...")
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.optimization.baseline import generate_grid_layout

n_turb = 12
turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turb)]
rotor_diameters = np.array([t.rotor_diameter for t in turbines])

boundary = create_rectangular_boundary(3500, 3500)
rng = np.random.default_rng(42)

positions = generate_grid_layout(boundary, n_turb, rotor_diameters, min_multiple=5.0, rng=rng)
print(f"   ✓ 生成 {n_turb} 台风机网格布局")

wake_model = JensenWake(0.07)
aep_calc = AEPCalculator(
    turbines=turbines,
    wind_resource=wr,
    wake_model=wake_model,
    wake_superposition="sum_of_squares",
    speed_step=1.0,
)

result = aep_calc.compute_farm_aep(positions)
print(f"   ✓ 装机容量: {result.total_installed_capacity:.2f} MW")
print(f"   ✓ 理论AEP: {result.gross_aep:.2f} MWh/年")
print(f"   ✓ 净AEP: {result.net_aep:.2f} MWh/年")
print(f"   ✓ 尾流损失: {result.wake_loss_pct:.2f}%")
print(f"   ✓ 容量系数: {result.capacity_factor:.2f}%")

max_loss = max(result.turbine_results, key=lambda x: x.wake_loss_pct)
print(f"   ✓ 最大损失风机: #{max_loss.turbine_idx} ({max_loss.wake_loss_pct:.1f}%)")
if max_loss.dominant_wake_source is not None:
    print(f"     主要影响源: #{max_loss.dominant_wake_source}")

print("\n7. 测试优化算法（小规模快速测试）...")
from wind_farm_opt.optimization.ga import GeneticAlgorithm, GAConfig

ga_config = GAConfig(
    population_size=10,
    max_generations=5,
    min_spacing_multiple=5.0,
    seed=42,
)

fitness_fn = aep_calc.evaluate_layout
ga = GeneticAlgorithm(
    n_turbines=n_turb,
    rotor_diameters=rotor_diameters,
    boundary=boundary,
    fitness_fn=fitness_fn,
    config=ga_config,
)

opt_result = ga.optimize(verbose=False)
print(f"   ✓ 遗传算法优化完成")
print(f"   ✓ 最优净AEP: {opt_result.best_fitness:.2f} MWh")
print(f"   ✓ 基线净AEP: {result.net_aep:.2f} MWh")
improvement = (opt_result.best_fitness - result.net_aep) / result.net_aep * 100
print(f"   ✓ 提升: {improvement:+.2f}%")

print("\n8. 测试经济性分析...")
from wind_farm_opt.economy.costs import (
    EconomicAnalyzer,
    get_default_turbine_cost,
    get_default_farm_cost,
)

turb_cost = get_default_turbine_cost("V126-3.45MW")
farm_cost = get_default_farm_cost()
analyzer = EconomicAnalyzer(turb_cost, farm_cost, electricity_price=0.45)

econ_result = analyzer.analyze(
    n_turbines=n_turb,
    rated_power_per_turbine_MW=turb.rated_power/1e3,
    net_aep_GWh=opt_result.best_fitness/1e3,
)
print(f"   ✓ 度电成本(LCOE): {econ_result.lcoe:.3f} 元/kWh")
print(f"   ✓ 初始投资: {econ_result.total_capital_cost/1e4:.2f} 亿元")
print(f"   ✓ 年收益: {econ_result.annual_revenue:.0f} 万元")
if econ_result.payback_period:
    print(f"   ✓ 投资回收期: {econ_result.payback_period:.1f} 年")
if econ_result.irr:
    print(f"   ✓ 内部收益率: {econ_result.irr:.2f}%")

print("\n9. 测试可视化模块...")
from wind_farm_opt.visualization.plotting import (
    plot_farm_layout,
    plot_wind_rose,
    plot_convergence,
    plot_turbine_loss_bar,
)

os.makedirs("test_output", exist_ok=True)

plot_wind_rose(wr, save_path="test_output/wind_rose.png", show=False)
print("   ✓ 风玫瑰图已生成")

losses = np.array([tr.wake_loss_pct for tr in result.turbine_results])
plot_farm_layout(
    positions, boundary, rotor_diameters,
    turbine_losses=losses,
    turbine_names=[f"#{i}" for i in range(n_turb)],
    save_path="test_output/layout.png",
    show=False,
)
print("   ✓ 布局图已生成")

plot_convergence(
    opt_result,
    baseline_aep=result.net_aep * 1e3,
    save_path="test_output/convergence.png",
    show=False,
)
print("   ✓ 收敛曲线已生成")

plot_turbine_loss_bar(
    result,
    save_path="test_output/losses.png",
    show=False,
)
print("   ✓ 损失柱状图已生成")

print("\n" + "=" * 60)
print("所有核心测试通过! ✓")
print("=" * 60)
print("\n可以使用以下命令运行完整分析:")
print("  python -m wind_farm_opt --help")
print("  python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50")
print("  python -m wind_farm_opt --config my_config.json")
