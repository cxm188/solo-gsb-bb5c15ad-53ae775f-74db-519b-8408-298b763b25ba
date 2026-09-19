"""纯计算逻辑：异常判定、校准偏差、线性插值与分段线性积分。不触碰数据库，便于单测。"""
from __future__ import annotations

from statistics import median


class CalibrationError(ValueError):
    """输入读数无法计算有效校准偏差时抛出。"""


def split_outliers(readings: list[float], tolerance: float) -> tuple[float, list[float], list[float]]:
    """按“偏离组中位数超过容差即异常”拆分读数。

    返回 (组中位数, 有效读数, 异常读数)。
    """
    group_median = median(readings)
    inliers = [r for r in readings if abs(r - group_median) <= tolerance]
    outliers = [r for r in readings if abs(r - group_median) > tolerance]
    return group_median, inliers, outliers


def compute_bias(reference: float, tolerance: float, readings: list[float]) -> tuple[float, float, list[float], list[float]]:
    """校准偏差 = 参考值 - 有效读数（剔除异常后）的中位数。

    返回 (bias, 组中位数, 有效读数, 异常读数)。
    """
    if len(readings) < 3:
        raise CalibrationError(f"need at least 3 readings, got {len(readings)}")
    group_median, inliers, outliers = split_outliers(readings, tolerance)
    if not inliers:
        raise CalibrationError(
            "all readings deviate from the group median by more than the tolerance; "
            "cannot compute a calibration bias"
        )
    bias = reference - median(inliers)
    return bias, group_median, inliers, outliers


def interpolate(t: float, t0: float, b0: float, t1: float, b1: float) -> float:
    """在相邻时点 (t0, b0) 与 (t1, b1) 之间对 t 做线性插值。"""
    if t1 == t0:
        return b0
    return b0 + (b1 - b0) * (t - t0) / (t1 - t0)


def exceedance_span(t0: float, b0: float, t1: float, b1: float, threshold: float) -> tuple[float, float] | None:
    """线性段 (t0,b0)-(t1,b1) 内 bias > threshold 的连续子区间 [start, end]，无则 None。

    恰一端越限时，另一端为解析交点 t* = t0 + (threshold − b0)·(t1 − t0)/(b1 − b0)。
    恰好等于阈值不计入超限：交点本身不属于超限集，但它是零测度端点，仍作为闭区间
    端点返回（不影响持续时长）。求 bias < −limit 的区间时，传入取负的端点偏差与
    threshold=limit 即可。全程解析解，不做固定步长采样。
    """
    above0 = b0 > threshold
    above1 = b1 > threshold
    if above0 and above1:
        return t0, t1
    if not above0 and not above1:
        return None
    t_star = t0 + (threshold - b0) * (t1 - t0) / (b1 - b0)
    return (t0, t_star) if above0 else (t_star, t1)


def integrate_segment(t0: float, b0: float, t1: float, b1: float) -> tuple[float, float]:
    """对线性段 (t0, b0)-(t1, b1) 下的面积做解析积分，返回 (signed, absolute)。

    signed 为梯形面积 (b0+b1)/2·Δt。absolute 为 |bias| 的面积：端点严格异号时
    段内恰有一个零点 t* = t0 − b0·Δt/(b1−b0)，在 t* 处拆成两个三角形求和
    （面积为 (b0² + b1²)·Δt / (2·|b1−b0|)）；否则 |bias| 不变号，面积即 |signed|。
    全程解析解，不做固定步长采样。
    """
    dt = t1 - t0
    if dt <= 0:
        return 0.0, 0.0
    signed = (b0 + b1) * 0.5 * dt
    if b0 * b1 < 0:  # 严格异号：段内恰有一个零点，按解析交点拆分
        absolute = (b0 * b0 + b1 * b1) * 0.5 * dt / abs(b1 - b0)
    else:  # 同号或端点触零：不变号
        absolute = abs(signed)
    return signed, absolute
