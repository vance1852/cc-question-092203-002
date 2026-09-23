"""尾流模型物理公式的回归测试。"""

import numpy as np

from wind_farm_opt.core.wake import JensenWake, superpose_wakes


def test_jensen_deficit_decays_downstream():
    """Jensen/PARK 顶帽亏损必须随下游距离单调衰减，且远场小于近场。

    回归保护：历史实现误用 ``1 - sqrt(1-Ct)·ratio^2``，
    使亏损随距离*增大*并趋于 1，远场风机被算成零风速、零发电量。
    """
    model = JensenWake(wake_decay=0.07)
    d = 126.0
    ct = 0.82
    distances = np.array([2.0, 5.0, 10.0, 20.0]) * d

    deficits = model.velocity_deficit(distances, d, ct)

    assert np.all(np.diff(deficits) < 0.0), deficits
    # 近场（2D）亏损约 0.26，远场（20D）应显著更小
    assert deficits[0] < 0.5
    assert deficits[-1] < deficits[0] * 0.5
    # 非正距离处无亏损
    assert model.velocity_deficit(0.0, d, ct) == 0.0


def test_superpose_sum_of_squares_bounds():
    combined = superpose_wakes(np.array([0.3, 0.3, 0.3]), method="sum_of_squares")
    assert np.isclose(combined, np.sqrt(3.0) * 0.3)
    # 线性叠加超过 1 时裁剪到 1
    assert superpose_wakes(np.array([0.9, 0.9]), method="linear") == 1.0
