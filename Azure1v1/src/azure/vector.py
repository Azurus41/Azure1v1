"""Small, dependency-free 3D vector used by the deterministic core."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]

    def __add__(self, other: "Vec3 | Iterable[float]") -> "Vec3":
        x, y, z = _coerce(other)
        return Vec3(self.x + x, self.y + y, self.z + z)

    __radd__ = __add__

    def __sub__(self, other: "Vec3 | Iterable[float]") -> "Vec3":
        x, y, z = _coerce(other)
        return Vec3(self.x - x, self.y - y, self.z - z)

    def __rsub__(self, other: "Vec3 | Iterable[float]") -> "Vec3":
        x, y, z = _coerce(other)
        return Vec3(x - self.x, y - self.y, z - self.z)

    def __mul__(self, other: float | "Vec3") -> "Vec3":
        if isinstance(other, (int, float)):
            return Vec3(self.x * other, self.y * other, self.z * other)
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def flat_length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec3":
        length = self.length()
        return self / length if length > 1e-9 else Vec3()

    def flat(self, normal: "Vec3 | None" = None) -> "Vec3":
        normal = normal or Vec3(0, 0, 1)
        return self - normal * self.dot(normal)

    def distance(self, other: "Vec3") -> float:
        return (self - other).length()

    def flat_distance(self, other: "Vec3", normal: "Vec3 | None" = None) -> float:
        return (self - other).flat(normal).flat_length()

    def scale(self, minimum: float = 0.0, maximum: float = 1.0) -> "Vec3":
        return self.normalized() * min(max(self.length(), minimum), maximum)

    def angle(self, other: "Vec3") -> float:
        denominator = self.length() * other.length()
        if denominator <= 1e-9:
            return 0.0
        return math.acos(max(-1.0, min(1.0, self.dot(other) / denominator)))


def vec(value: "Vec3 | Iterable[float] | None") -> Vec3:
    if value is None:
        return Vec3()
    if isinstance(value, Vec3):
        return value
    return Vec3(*_coerce(value))


def _coerce(value: "Vec3 | Iterable[float]") -> tuple[float, float, float]:
    if isinstance(value, Vec3):
        return value.x, value.y, value.z
    if all(hasattr(value, component) for component in ("x", "y", "z")):
        return float(value.x), float(value.y), float(value.z)
    values = tuple(float(component) for component in value)
    if len(values) != 3:
        raise ValueError(f"Expected 3 components, got {len(values)}")
    return values  # type: ignore[return-value]
