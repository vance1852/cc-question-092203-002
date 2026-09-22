"""风机间距约束。"""

import numpy as np


def check_min_spacing(
    positions: np.ndarray,
    min_distance: float,
) -> tuple[bool, np.ndarray]:
    """检查所有风机对之间的间距是否满足最小距离要求。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N_turbines, 2)
    min_distance : float
        最小允许间距 (m)

    Returns
    -------
    tuple[bool, np.ndarray]
        - 是否所有间距都满足要求
        - 不满足要求的风机对索引数组，形状为 (M, 2)，M 为违规对数
    """
    n = positions.shape[0]
    violations = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_distance:
                violations.append([i, j])

    if violations:
        return False, np.array(violations, dtype=int)
    else:
        return True, np.zeros((0, 2), dtype=int)


def compute_min_spacing_from_diameters(
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
) -> float:
    """根据转子直径计算最小间距（取最大直径的倍数）。

    Parameters
    ----------
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数（相对于转子直径）

    Returns
    -------
    float
        最小间距 (m)
    """
    return float(min_multiple * np.max(rotor_diameters))


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """计算所有风机对之间的距离矩阵。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N, 2)

    Returns
    -------
    np.ndarray
        距离矩阵，形状为 (N, N)，对角线为 0
    """
    n = positions.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def enforce_min_spacing(
    positions: np.ndarray,
    min_distance: float,
    boundary,
    rng: np.random.Generator | None = None,
    max_iterations: int = 1000,
) -> np.ndarray:
    """尝试通过移动风机来满足最小间距约束。

    当有风机对间距不足时，将它们沿连线方向推开。

    Parameters
    ----------
    positions : np.ndarray
        初始风机位置，形状为 (N, 2)
    min_distance : float
        最小间距 (m)
    boundary : SiteBoundary
        场地边界
    rng : Optional[np.random.Generator]
        随机数生成器
    max_iterations : int
        最大迭代次数

    Returns
    -------
    np.ndarray
        调整后的风机位置
    """
    if rng is None:
        rng = np.random.default_rng()

    positions = positions.copy()
    n = positions.shape[0]

    for _ in range(max_iterations):
        valid, violations = check_min_spacing(positions, min_distance)
        if valid:
            break

        for i, j in violations:
            vec = positions[j] - positions[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-12:
                vec = rng.standard_normal(2)
                dist = np.linalg.norm(vec)
            vec_norm = vec / dist

            push = (min_distance - dist) / 2.0 + 1e-6
            positions[i] -= vec_norm * push
            positions[j] += vec_norm * push

        for k in range(n):
            if not boundary.contains_point(positions[k]):
                positions[k] = boundary.project_to_boundary(positions[k])
                perturbation = rng.uniform(-5.0, 5.0, 2)
                positions[k] += perturbation
                if not boundary.contains_point(positions[k]):
                    positions[k] = boundary.project_to_boundary(positions[k])

    valid, _ = check_min_spacing(positions, min_distance)
    inside = boundary.contains_all(positions)
    if not (valid and inside.all()):
        raise RuntimeError("无法通过调整满足间距和边界约束")

    return positions
