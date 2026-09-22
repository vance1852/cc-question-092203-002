"""风资源模型。"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _gamma_lanczos(x: np.ndarray) -> np.ndarray:
    """Lanczos 近似的向量化 gamma 函数。

    使用 Lanczos 公式计算 gamma(x)，适用于 x >= 0.5。
    系数取自 GSL 实现，g=7，n=9。

    Parameters
    ----------
    x : np.ndarray
        输入数组，所有元素必须 >= 0.5

    Returns
    -------
    np.ndarray
        gamma(x) 的值
    """
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x[None]

    g = 7.0
    coeffs = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ])

    n = len(coeffs)
    x1 = x - 1.0
    t = x1 + g + 0.5

    denom = x1[None, :] + np.arange(1, n)[:, None]
    terms = coeffs[1:, None] / denom
    series = coeffs[0] + np.sum(terms, axis=0)

    result = np.sqrt(2 * np.pi) * t ** (x1 + 0.5) * np.exp(-t) * series

    if scalar_input:
        result = result[0]

    return result


@dataclass
class WindSector:
    """单个风向扇区的数据。

    Parameters
    ----------
    direction_center : float
        扇区中心方向 (度)，0度为正北，顺时针
    direction_width : float
        扇区宽度 (度)
    frequency : float
        该扇区的风频率 (0-1)
    mean_speed : float
        平均风速 (m/s)
    weibull_k : float
        威布尔形状参数 k
    weibull_c : float
        威布尔尺度参数 c (m/s)
    """

    direction_center: float
    direction_width: float
    frequency: float
    mean_speed: float
    weibull_k: float
    weibull_c: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.frequency <= 1.0):
            raise ValueError(f"频率必须在 [0, 1] 范围内，当前为 {self.frequency}")
        if self.mean_speed <= 0:
            raise ValueError(f"平均风速必须大于0，当前为 {self.mean_speed}")
        if self.weibull_k <= 0:
            raise ValueError(f"威布尔k参数必须大于0，当前为 {self.weibull_k}")
        if self.weibull_c <= 0:
            raise ValueError(f"威布尔c参数必须大于0，当前为 {self.weibull_c}")


class WindResource:
    """风资源数据类。

    包含按风向扇区划分的风玫瑰数据。
    """

    def __init__(self, sectors: list[WindSector]) -> None:
        self.sectors = sectors
        self._validate()

    def _validate(self) -> None:
        """验证扇区数据完整性。"""
        total_freq = sum(s.frequency for s in self.sectors)
        if abs(total_freq - 1.0) > 1e-6:
            raise ValueError(
                f"所有扇区频率之和应为1.0，当前为 {total_freq:.6f}"
            )

    @property
    def num_sectors(self) -> int:
        """扇区数量。"""
        return len(self.sectors)

    @property
    def directions(self) -> np.ndarray:
        """所有扇区中心方向数组 (度)。"""
        return np.array([s.direction_center for s in self.sectors], dtype=np.float64)

    @property
    def frequencies(self) -> np.ndarray:
        """所有扇区频率数组。"""
        return np.array([s.frequency for s in self.sectors], dtype=np.float64)

    @property
    def mean_speeds(self) -> np.ndarray:
        """所有扇区平均风速数组 (m/s)。"""
        return np.array([s.mean_speed for s in self.sectors], dtype=np.float64)

    @property
    def weibull_ks(self) -> np.ndarray:
        """所有扇区威布尔k参数数组。"""
        return np.array([s.weibull_k for s in self.sectors], dtype=np.float64)

    @property
    def weibull_cs(self) -> np.ndarray:
        """所有扇区威布尔c参数数组 (m/s)。"""
        return np.array([s.weibull_c for s in self.sectors], dtype=np.float64)

    @property
    def sector_widths(self) -> np.ndarray:
        """所有扇区宽度数组 (度)。"""
        return np.array([s.direction_width for s in self.sectors], dtype=np.float64)

    def weibull_pdf(self, wind_speed: np.ndarray, sector_idx: int) -> np.ndarray:
        """计算指定扇区的威布尔概率密度函数。

        Parameters
        ----------
        wind_speed : np.ndarray
            风速数组 (m/s)
        sector_idx : int
            扇区索引

        Returns
        -------
        np.ndarray
            概率密度值
        """
        k = self.sectors[sector_idx].weibull_k
        c = self.sectors[sector_idx].weibull_c
        ws = np.asarray(wind_speed, dtype=np.float64)
        return (k / c) * (ws / c) ** (k - 1) * np.exp(-(ws / c) ** k)

    def weibull_mean(self, sector_idx: int) -> float:
        """计算指定扇区的威布尔分布均值。

        Parameters
        ----------
        sector_idx : int
            扇区索引

        Returns
        -------
        float
            均值风速 (m/s)
        """
        k = self.sectors[sector_idx].weibull_k
        c = self.sectors[sector_idx].weibull_c
        return float(c * _gamma_lanczos(1.0 + 1.0 / k))

    @property
    def overall_mean_speed(self) -> float:
        """全场加权平均风速 (m/s)。"""
        return float(np.sum(self.frequencies * self.mean_speeds))


def create_default_wind_resource(
    num_sectors: int = 12,
    dominant_direction: float = 270.0,
    mean_speed: float = 8.5,
) -> WindResource:
    """创建默认的风资源数据（内置风玫瑰）。

    创建一个12扇区（每30度一个）的风玫瑰，主风向可指定。

    Parameters
    ----------
    num_sectors : int
        扇区数量，默认为12（每30度）
    dominant_direction : float
        主风向 (度)，默认西风（270度）
    mean_speed : float
        平均风速基准值 (m/s)

    Returns
    -------
    WindResource
        风资源实例
    """
    width = 360.0 / num_sectors
    centers = np.arange(width / 2.0, 360.0, width)

    angles = np.deg2rad(centers - dominant_direction)
    freq = 0.5 * (1.0 + 0.7 * np.cos(angles))
    freq = freq / freq.sum()

    speed_variation = 0.7 + 0.3 * np.cos(angles)
    speeds = mean_speed * speed_variation

    k_values = np.full(num_sectors, 2.1)

    c_values = speeds / _gamma_lanczos(1.0 + 1.0 / k_values)

    sectors = []
    for i in range(num_sectors):
        sectors.append(
            WindSector(
                direction_center=float(centers[i]),
                direction_width=float(width),
                frequency=float(freq[i]),
                mean_speed=float(speeds[i]),
                weibull_k=float(k_values[i]),
                weibull_c=float(c_values[i]),
            )
        )

    return WindResource(sectors)


def create_simple_wind_resource(
    num_sectors: int = 12,
    uniform: bool = False,
    mean_speed: float = 8.0,
) -> WindResource:
    """创建简化的风资源数据。

    Parameters
    ----------
    num_sectors : int
        扇区数量
    uniform : bool
        是否均匀分布各风向
    mean_speed : float
        平均风速 (m/s)

    Returns
    -------
    WindResource
        风资源实例
    """
    if uniform:
        width = 360.0 / num_sectors
        centers = np.arange(width / 2.0, 360.0, width)
        freq = np.full(num_sectors, 1.0 / num_sectors)
        speeds = np.full(num_sectors, mean_speed)
    else:
        dominant_dirs = [45.0, 225.0]
        dominant_weights = [0.4, 0.4]
        return _create_multimodal_wind(num_sectors, dominant_dirs, dominant_weights, mean_speed)

    k_values = np.full(num_sectors, 2.0)
    c_values = speeds / _gamma_lanczos(1.0 + 1.0 / k_values)
    width = 360.0 / num_sectors
    centers = np.arange(width / 2.0, 360.0, width)

    sectors = []
    for i in range(num_sectors):
        sectors.append(
            WindSector(
                direction_center=float(centers[i]),
                direction_width=float(width),
                frequency=float(freq[i]),
                mean_speed=float(speeds[i]),
                weibull_k=float(k_values[i]),
                weibull_c=float(c_values[i]),
            )
        )

    return WindResource(sectors)


def _create_multimodal_wind(
    num_sectors: int,
    dominant_dirs: list[float],
    dominant_weights: list[float],
    mean_speed: float,
) -> WindResource:
    width = 360.0 / num_sectors
    centers = np.arange(width / 2.0, 360.0, width)

    freq = np.full(num_sectors, (1.0 - sum(dominant_weights)) / num_sectors)

    sigma = np.deg2rad(30.0)
    for dir_center, weight in zip(dominant_dirs, dominant_weights):
        angles = np.deg2rad(centers - dir_center)
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        gaussian = np.exp(-0.5 * (angles / sigma) ** 2)
        gaussian = gaussian / gaussian.sum() * weight
        freq += gaussian

    freq = freq / freq.sum()

    speeds = np.full(num_sectors, mean_speed)
    k_values = np.full(num_sectors, 2.0)

    c_values = speeds / _gamma_lanczos(1.0 + 1.0 / k_values)

    sectors = []
    for i in range(num_sectors):
        sectors.append(
            WindSector(
                direction_center=float(centers[i]),
                direction_width=float(width),
                frequency=float(freq[i]),
                mean_speed=float(speeds[i]),
                weibull_k=float(k_values[i]),
                weibull_c=float(c_values[i]),
            )
        )

    return WindResource(sectors)
