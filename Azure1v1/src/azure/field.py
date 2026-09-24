"""Standard arena geometry, boost state, and target geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .constants import (
    BALL_RADIUS, FIELD_HALF_LENGTH, FIELD_HALF_WIDTH, FIELD_HEIGHT,
    GOAL_CENTER_Z, GOAL_DEPTH, GOAL_HEIGHT, GOAL_WIDTH,
)
from .vector import Vec3

SQRT_HALF = math.sqrt(0.5)


@dataclass(frozen=True, slots=True)
class Surface:
    name: str
    center: Vec3
    normal: Vec3
    half_size: Vec3
    x_axis: Vec3
    y_axis: Vec3
    drivable: bool = True

    def project(self, point: Vec3) -> Vec3:
        offset = point - self.center
        return self.center + (
            self.x_axis * _clamp(self.x_axis.dot(offset), -self.half_size.x, self.half_size.x)
            + self.y_axis * _clamp(self.y_axis.dot(offset), -self.half_size.y, self.half_size.y)
        )

    def height_above(self, point: Vec3) -> float:
        return (point - self.project(point)).dot(self.normal)


@dataclass(frozen=True, slots=True)
class BoostPad:
    index: int
    location: Vec3
    is_large: bool
    is_active: bool = True
    timer: float = 0.0

    @property
    def amount(self) -> float:
        return 100.0 if self.is_large else 12.0


def make_standard_surfaces() -> tuple[Surface, ...]:
    return (
        Surface("ground", Vec3(), Vec3(0, 0, 1), Vec3(FIELD_HALF_WIDTH, FIELD_HALF_LENGTH), Vec3(1, 0, 0), Vec3(0, 1, 0)),
        Surface("ceiling", Vec3(0, 0, FIELD_HEIGHT), Vec3(0, 0, -1), Vec3(FIELD_HALF_WIDTH, FIELD_HALF_LENGTH), Vec3(1, 0, 0), Vec3(0, 1, 0), False),
        Surface("right wall", Vec3(FIELD_HALF_WIDTH, 0, FIELD_HEIGHT / 2), Vec3(-1, 0, 0), Vec3(FIELD_HEIGHT / 2, FIELD_HALF_LENGTH), Vec3(0, 0, 1), Vec3(0, 1, 0)),
        Surface("left wall", Vec3(-FIELD_HALF_WIDTH, 0, FIELD_HEIGHT / 2), Vec3(1, 0, 0), Vec3(FIELD_HEIGHT / 2, FIELD_HALF_LENGTH), Vec3(0, 0, 1), Vec3(0, 1, 0)),
        Surface("orange backboard", Vec3(0, FIELD_HALF_LENGTH, (FIELD_HEIGHT + GOAL_HEIGHT) / 2), Vec3(0, -1, 0), Vec3(GOAL_WIDTH / 2, (FIELD_HEIGHT - GOAL_HEIGHT) / 2), Vec3(1, 0, 0), Vec3(0, 0, 1)),
        Surface("blue backboard", Vec3(0, -FIELD_HALF_LENGTH, (FIELD_HEIGHT + GOAL_HEIGHT) / 2), Vec3(0, 1, 0), Vec3(GOAL_WIDTH / 2, (FIELD_HEIGHT - GOAL_HEIGHT) / 2), Vec3(1, 0, 0), Vec3(0, 0, 1)),
        _corner("orange right", FIELD_HALF_WIDTH, FIELD_HALF_LENGTH, -SQRT_HALF, -SQRT_HALF, -SQRT_HALF, SQRT_HALF),
        _corner("orange left", -FIELD_HALF_WIDTH, FIELD_HALF_LENGTH, SQRT_HALF, -SQRT_HALF, SQRT_HALF, SQRT_HALF),
        _corner("blue right", FIELD_HALF_WIDTH, -FIELD_HALF_LENGTH, -SQRT_HALF, SQRT_HALF, -SQRT_HALF, -SQRT_HALF),
        _corner("blue left", -FIELD_HALF_WIDTH, -FIELD_HALF_LENGTH, SQRT_HALF, SQRT_HALF, SQRT_HALF, -SQRT_HALF),
        Surface("orange goal floor", Vec3(0, FIELD_HALF_LENGTH + GOAL_DEPTH / 2, 0), Vec3(0, 0, 1), Vec3(GOAL_WIDTH / 2, GOAL_DEPTH / 2), Vec3(1, 0, 0), Vec3(0, 1, 0)),
        Surface("blue goal floor", Vec3(0, -FIELD_HALF_LENGTH - GOAL_DEPTH / 2, 0), Vec3(0, 0, 1), Vec3(GOAL_WIDTH / 2, GOAL_DEPTH / 2), Vec3(1, 0, 0), Vec3(0, 1, 0)),
    )


def nearest_surface(point: Vec3, *, include_ceiling: bool = True) -> Surface:
    candidates = STANDARD_SURFACES
    if not include_ceiling:
        candidates = tuple(surface for surface in candidates if surface.name != "ceiling")
    return min(candidates, key=lambda surface: (point - surface.project(point)).length())


def in_field(point: Vec3, margin: float = 0.0) -> bool:
    return (
        abs(point.x) <= FIELD_HALF_WIDTH - margin
        and abs(point.y) <= FIELD_HALF_LENGTH - margin
        and margin <= point.z <= FIELD_HEIGHT - margin
    )


def team_attack_direction(team: int) -> float:
    return 1.0 if team == 0 else -1.0


def own_goal(team: int) -> Vec3:
    return Vec3(0, -team_attack_direction(team) * FIELD_HALF_LENGTH, GOAL_CENTER_Z)


def opponent_goal(team: int) -> Vec3:
    return Vec3(0, team_attack_direction(team) * FIELD_HALF_LENGTH, GOAL_CENTER_Z)


def goal_target(team: int, *, away: bool = False) -> Vec3:
    direction = team_attack_direction(team)
    x = 400.0 if (away == (team == 1)) else -400.0
    return Vec3(x, direction * FIELD_HALF_LENGTH, 250.0)


def clear_target(location: Vec3, team: int) -> Vec3:
    """Choose a safe clear away from our goal and toward an opponent corner."""
    attack_y = team_attack_direction(team)
    own_y = -attack_y * FIELD_HALF_LENGTH
    away = (location - Vec3(0, own_y, 0)).flat().normalized()
    if away.length() < 0.1:
        away = Vec3(0, attack_y, 0)
    side = 1.0 if location.x >= 0 else -1.0
    if away.dot(Vec3(0, attack_y, 0)) > -0.1:
        return Vec3(side * 3100.0, attack_y * 3900.0, 250.0)
    return Vec3(side * 2500.0, own_y + attack_y * 900.0, 180.0)


def point_is_in_goal(point: Vec3, team: int) -> bool:
    defended_direction = -team_attack_direction(team)
    return (
        abs(point.x) < GOAL_WIDTH / 2 + BALL_RADIUS
        and point.y * defended_direction > FIELD_HALF_LENGTH - BALL_RADIUS
        and 0 <= point.z <= GOAL_HEIGHT + BALL_RADIUS
    )


def clamp_target(point: Vec3, team: int, radius: float = BALL_RADIUS + 15.0) -> Vec3:
    """Clamp an on-goal point to a safe scoring target behind the goal line."""
    direction = team_attack_direction(team)
    return Vec3(
        _clamp(point.x, -GOAL_WIDTH / 2 + radius, GOAL_WIDTH / 2 - radius),
        direction * (FIELD_HALF_LENGTH + BALL_RADIUS + 25.0),
        _clamp(point.z, BALL_RADIUS + 20.0, GOAL_HEIGHT - BALL_RADIUS - 20.0),
    )


def _corner(name: str, x: float, y: float, nx: float, ny: float, xx: float, xy: float) -> Surface:
    center = Vec3(x - math.copysign(576.0, x), y - math.copysign(576.0, y), FIELD_HEIGHT / 2)
    return Surface(name, center, Vec3(nx, ny, 0), Vec3(814.5, FIELD_HEIGHT / 2), Vec3(xx, xy, 0), Vec3(0, 0, 1))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


STANDARD_SURFACES = make_standard_surfaces()
