"""基线布局生成（规则网格）。"""

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    enforce_min_spacing,
)


def generate_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    aspect_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成规则网格布局作为优化基线。

    Parameters
    ----------
    boundary : SiteBoundary
        场地边界
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数
    aspect_ratio : float
        网格纵横比 (列数/行数)
    rng : Optional[np.random.Generator]
        随机数生成器

    Returns
    -------
    np.ndarray
        网格布局位置 (n_turbines, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    min_spacing = compute_min_spacing_from_diameters(rotor_diameters, min_multiple)

    n_rows = max(1, int(np.round(np.sqrt(n_turbines / aspect_ratio))))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    x_min, x_max = boundary.x_min, boundary.x_max
    y_min, y_max = boundary.y_min, boundary.y_max

    margin = min_spacing * 0.5
    x_range = x_max - x_min - 2 * margin
    y_range = y_max - y_min - 2 * margin

    spacing_x = min(x_range / max(n_cols - 1, 1), min_spacing * 1.5)
    spacing_y = min(y_range / max(n_rows - 1, 1), min_spacing * 1.5)

    positions = []

    start_x = x_min + margin + (x_range - spacing_x * (n_cols - 1)) / 2.0
    start_y = y_min + margin + (y_range - spacing_y * (n_rows - 1)) / 2.0

    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= n_turbines:
                break
            x = start_x + col * spacing_x
            y = start_y + row * spacing_y
            pos = np.array([x, y])
            if boundary.contains_point(pos):
                positions.append(pos)
                count += 1

    while len(positions) < n_turbines:
        candidates = boundary.sample_random_points(n_turbines * 2, rng)
        for cand in candidates:
            if len(positions) >= n_turbines:
                break
            pos_arr = np.array(positions + [cand]) if positions else np.array([cand])
            valid, _ = check_min_spacing(pos_arr, min_spacing)
            if valid and boundary.contains_point(cand):
                positions.append(cand)

    positions = np.array(positions, dtype=np.float64)

    valid, _ = check_min_spacing(positions, min_spacing)
    inside = boundary.contains_all(positions).all()

    if not (valid and inside):
        try:
            positions = enforce_min_spacing(positions, min_spacing, boundary, rng)
        except RuntimeError:
            pass

    return positions


def generate_staggered_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    dominant_direction: float = 270.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成交错网格布局（错位排列，减少主风向下的尾流）。

    Parameters
    ----------
    boundary : SiteBoundary
        场地边界
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数
    dominant_direction : float
        主风向（度），用于确定交错方向
    rng : Optional[np.random.Generator]
        随机数生成器

    Returns
    -------
    np.ndarray
        交错网格布局位置 (n_turbines, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    min_spacing = compute_min_spacing_from_diameters(rotor_diameters, min_multiple)

    n_rows = max(1, int(np.sqrt(n_turbines)))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    x_min, x_max = boundary.x_min, boundary.x_max
    y_min, y_max = boundary.y_min, boundary.y_max

    margin = min_spacing * 0.5
    x_range = x_max - x_min - 2 * margin
    y_range = y_max - y_min - 2 * margin

    spacing_x = max(x_range / max(n_cols - 1, 1), min_spacing * 1.2)
    spacing_y = max(y_range / max(n_rows - 1, 1), min_spacing * 1.2)

    positions = []

    start_x = x_min + margin + (x_range - spacing_x * (n_cols - 1)) / 2.0
    start_y = y_min + margin + (y_range - spacing_y * (n_rows - 1)) / 2.0

    count = 0
    for row in range(n_rows):
        offset = spacing_x / 2.0 if row % 2 == 1 else 0.0
        for col in range(n_cols):
            if count >= n_turbines:
                break
            x = start_x + col * spacing_x + offset
            y = start_y + row * spacing_y
            pos = np.array([x, y])
            if boundary.contains_point(pos):
                positions.append(pos)
                count += 1

    while len(positions) < n_turbines:
        candidates = boundary.sample_random_points(n_turbines * 2, rng)
        for cand in candidates:
            if len(positions) >= n_turbines:
                break
            pos_arr = np.array(positions + [cand]) if positions else np.array([cand])
            valid, _ = check_min_spacing(pos_arr, min_spacing)
            if valid and boundary.contains_point(cand):
                positions.append(cand)

    positions = np.array(positions, dtype=np.float64)

    valid, _ = check_min_spacing(positions, min_spacing)
    inside = boundary.contains_all(positions).all()

    if not (valid and inside):
        try:
            positions = enforce_min_spacing(positions, min_spacing, boundary, rng)
        except RuntimeError:
            pass

    return positions
