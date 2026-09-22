"""场地边界约束。

支持任意多边形边界，使用射线法判断点是否在多边形内。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SiteBoundary:
    """场地边界类。

    使用闭合多边形定义场地范围。

    Parameters
    ----------
    vertices : np.ndarray
        多边形顶点坐标，形状为 (N, 2)，单位为米。
        多边形会自动闭合，不需要重复起点。
    """

    vertices: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("顶点坐标必须是形状为 (N, 2) 的数组")
        if self.vertices.shape[0] < 3:
            raise ValueError("多边形至少需要3个顶点")

    @property
    def x_min(self) -> float:
        return float(np.min(self.vertices[:, 0]))

    @property
    def x_max(self) -> float:
        return float(np.max(self.vertices[:, 0]))

    @property
    def y_min(self) -> float:
        return float(np.min(self.vertices[:, 1]))

    @property
    def y_max(self) -> float:
        return float(np.max(self.vertices[:, 1]))

    @property
    def area(self) -> float:
        """使用 shoelace 公式计算多边形面积。"""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        n = len(x)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j] - x[j] * y[i]
        return float(abs(area) / 2.0)

    def contains_point(
        self,
        point: np.ndarray,
        tolerance: float = 1e-9,
    ) -> bool:
        """判断点是否在多边形内部（射线法）。

        Parameters
        ----------
        point : np.ndarray
            点坐标，形状为 (2,)
        tolerance : float
            边界判定容差

        Returns
        -------
        bool
            True 表示点在多边形内部或边界上
        """
        pt = np.asarray(point, dtype=np.float64)
        verts = self.vertices

        if self._on_edge(pt, tolerance):
            return True

        n = len(verts)
        inside = False
        x, y = pt[0], pt[1]

        for i in range(n):
            j = (i + 1) % n
            xi, yi = verts[i]
            xj, yj = verts[j]

            if ((yi > y) != (yj > y)):
                x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x <= x_intersect + tolerance:
                    inside = not inside

        return inside

    def _on_edge(self, point: np.ndarray, tolerance: float) -> bool:
        """检查点是否在多边形边界上。"""
        verts = self.vertices
        n = len(verts)

        for i in range(n):
            j = (i + 1) % n
            if self._point_on_segment(point, verts[i], verts[j], tolerance):
                return True
        return False

    @staticmethod
    def _point_on_segment(
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
        tolerance: float,
    ) -> bool:
        """判断点是否在线段上。"""
        cross = (point[0] - seg_start[0]) * (seg_end[1] - seg_start[1]) - \
                (point[1] - seg_start[1]) * (seg_end[0] - seg_start[0])
        if abs(cross) > tolerance:
            return False

        dot = (point[0] - seg_start[0]) * (seg_end[0] - seg_start[0]) + \
              (point[1] - seg_start[1]) * (seg_end[1] - seg_start[1])
        if dot < -tolerance:
            return False

        len_sq = (seg_end[0] - seg_start[0]) ** 2 + (seg_end[1] - seg_start[1]) ** 2
        if dot > len_sq + tolerance:
            return False

        return True

    def contains_all(self, positions: np.ndarray) -> np.ndarray:
        """批量检查多个点是否在多边形内部。

        Parameters
        ----------
        positions : np.ndarray
            点坐标，形状为 (N, 2)

        Returns
        -------
        np.ndarray
            布尔数组，形状为 (N,)
        """
        positions = np.asarray(positions, dtype=np.float64)
        result = np.zeros(positions.shape[0], dtype=bool)
        for i, pt in enumerate(positions):
            result[i] = self.contains_point(pt)
        return result

    def project_to_boundary(self, point: np.ndarray) -> np.ndarray:
        """将点投影到多边形边界上（最近点）。

        Parameters
        ----------
        point : np.ndarray
            原始点坐标，形状为 (2,)

        Returns
        -------
        np.ndarray
            投影后的点坐标，形状为 (2,)
        """
        pt = np.asarray(point, dtype=np.float64)
        verts = self.vertices
        n = len(verts)

        best_dist = np.inf
        best_point = verts[0].copy()

        for i in range(n):
            j = (i + 1) % n
            proj = self._project_to_segment(pt, verts[i], verts[j])
            dist = np.linalg.norm(pt - proj)
            if dist < best_dist:
                best_dist = dist
                best_point = proj

        return best_point

    @staticmethod
    def _project_to_segment(
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
    ) -> np.ndarray:
        """将点投影到线段上。"""
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)

        if seg_len_sq < 1e-12:
            return seg_start.copy()

        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)

        return seg_start + t * seg_vec

    def sample_random_points(
        self,
        n_points: int,
        rng: Optional[np.random.Generator] = None,
        max_attempts: int = 100,
    ) -> np.ndarray:
        """在多边形内随机采样点（拒绝采样）。

        Parameters
        ----------
        n_points : int
            需要采样的点数
        rng : Optional[np.random.Generator]
            随机数生成器
        max_attempts : int
            每个点的最大尝试次数

        Returns
        -------
        np.ndarray
            采样点坐标，形状为 (n_points, 2)
        """
        if rng is None:
            rng = np.random.default_rng()

        points = np.zeros((n_points, 2), dtype=np.float64)
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        for i in range(n_points):
            found = False
            for _ in range(max_attempts):
                x = rng.uniform(x_min, x_max)
                y = rng.uniform(y_min, y_max)
                pt = np.array([x, y])
                if self.contains_point(pt):
                    points[i] = pt
                    found = True
                    break
            if not found:
                raise RuntimeError(f"无法在场地内采样到第 {i+1} 个点")

        return points


def create_rectangular_boundary(
    width: float,
    height: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> SiteBoundary:
    """创建矩形场地边界。

    Parameters
    ----------
    width : float
        宽度（x方向）(m)
    height : float
        高度（y方向）(m)
    center_x : float
        中心x坐标 (m)
    center_y : float
        中心y坐标 (m)

    Returns
    -------
    SiteBoundary
        矩形场地边界
    """
    x1 = center_x - width / 2.0
    x2 = center_x + width / 2.0
    y1 = center_y - height / 2.0
    y2 = center_y + height / 2.0

    vertices = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float64)

    return SiteBoundary(vertices)


def create_hexagonal_boundary(
    radius: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> SiteBoundary:
    """创建正六边形场地边界。

    Parameters
    ----------
    radius : float
        外接圆半径 (m)
    center_x : float
        中心x坐标 (m)
    center_y : float
        中心y坐标 (m)

    Returns
    -------
    SiteBoundary
        六边形场地边界
    """
    angles = np.deg2rad(np.arange(0, 360, 60))
    vertices = np.column_stack([
        center_x + radius * np.cos(angles),
        center_y + radius * np.sin(angles),
    ])
    return SiteBoundary(vertices)


def create_irregular_boundary() -> SiteBoundary:
    """创建一个不规则多边形场地边界作为示例。

    Returns
    -------
    SiteBoundary
        不规则场地边界
    """
    vertices = np.array([
        [0.0, 0.0],
        [3000.0, -200.0],
        [3200.0, 1500.0],
        [2800.0, 2800.0],
        [1500.0, 3000.0],
        [-200.0, 2500.0],
        [-300.0, 1200.0],
    ], dtype=np.float64)
    return SiteBoundary(vertices)
