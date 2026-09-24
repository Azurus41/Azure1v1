"""Ground and aerial control primitives derived from Party Canon."""

from __future__ import annotations

import math

from rlbot.flat import ControllerState

from .constants import BOOST_ACCEL, MAX_SPEED
from .model import CarState
from .vector import Vec3


def clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def cubic_pd(error: float, rate: float) -> float:
    return clamp((35.0 * (error + rate)) ** 3 / 10.0)


def aim_at(car: CarState, direction: Vec3, controls: ControllerState, desired_up: Vec3 = Vec3(0, 0, 1)) -> None:
    local_target = car.local(direction)
    if local_target.length() < 1e-6:
        return
    local_up = car.local(desired_up.normalized())
    pitch = math.atan2(local_target.z, local_target.x)
    yaw = math.atan2(local_target.y, local_target.x)
    roll = math.atan2(local_up.y, local_up.z)
    local_angular = car.local(car.angular_velocity)
    controls.steer = clamp(cubic_pd(yaw, -local_angular.z * 0.01))
    controls.yaw = clamp(cubic_pd(yaw, -local_angular.z * 0.15))
    controls.pitch = clamp(cubic_pd(pitch, local_angular.y * 0.20))
    controls.roll = clamp(cubic_pd(roll, local_angular.x * 0.25))


def throttle_to(car: CarState, target_speed: float, controls: ControllerState, *, backwards: bool = False, allow_boost: bool = True) -> float:
    forward_speed = car.velocity.dot(car.forward)
    desired = -abs(target_speed) if backwards else abs(target_speed)
    difference = desired - forward_speed
    controls.throttle = clamp(difference * abs(difference) / 1000.0)
    controls.boost = bool(
        allow_boost and not backwards and target_speed > 1400 and difference > 50
        and forward_speed < 2250 and controls.throttle == 1.0 and car.boost > 1
    )
    return forward_speed


def drive_to(
    car: CarState, target: Vec3, controls: ControllerState, *,
    target_speed: float = MAX_SPEED, allow_boost: bool = True,
    desired_up: Vec3 | None = None,
) -> None:
    """Party-style drive control with turn-radius-aware target shifting."""
    speed = car.velocity.length()
    if not car.grounded and desired_up is not None:
        aim_at(car, target - car.location, controls, desired_up)
        throttle_to(car, target_speed, controls, allow_boost=False)
        return
    normal = car.up if car.grounded else Vec3(0, 0, 1)
    flat_target = target.flat(normal)
    flat_car = car.location.flat(normal)
    delta = flat_target - flat_car
    distance = delta.length()
    turn_radius = turn_radius_at(speed)
    if distance > 200 and turn_radius < distance:
        right = car.right.flat(normal).normalized()
        sign = 1.0 if right.dot(delta) >= 0 else -1.0
        center = flat_car + right * turn_radius * sign
        remaining = max(0.0, (center - flat_target).flat(normal).length() - turn_radius)
        angle = math.acos(clamp((center - flat_target).flat(normal).normalized().dot(car.forward.flat(normal)), -1.0, 1.0))
        aim = flat_car - right * turn_radius * sign * (1 - math.cos(min(angle, math.pi)))
        aim += car.forward.flat(normal) * min(distance, math.sin(min(angle, math.pi)) * turn_radius)
        flat_target = aim + delta * (remaining / max(distance, 1.0))
    angle_to_target = math.atan2(delta.flat(normal).dot(car.right.flat(normal)), delta.dot(car.forward))
    speed_cap = min(target_speed, speed_from_radius(max(turn_radius, 145)))
    if abs(angle_to_target) < 0.35 and distance > turn_radius:
        speed_cap = target_speed
    aim_at(car, flat_target - car.location, controls, normal)
    throttle_to(car, speed_cap, controls, allow_boost=allow_boost and abs(angle_to_target) < 0.28 and car.up.dot(normal) > 0.9)



def aerial_to(
    car: CarState, target: Vec3, seconds: float, controls: ControllerState, *,
    desired_up: Vec3 = Vec3(0, 0, 1), boost: bool = True,
    delta: float = 1 / 120,
) -> None:
    predicted, _ = car.predict(seconds, 0.0)
    offset = target - predicted
    direction = offset.normalized()
    if direction.length() < 0.1:
        direction = car.forward
    aim_at(car, direction, controls, desired_up)
    projected_velocity = offset.dot(car.forward) / max(seconds, 1 / 120)
    controls.throttle = clamp(projected_velocity / (66.6666667 * max(delta, 1 / 120)))
    required_acceleration = projected_velocity / max(seconds, 1 / 120)
    controls.boost = bool(
        boost and car.boost > 0 and direction.angle(car.forward) < 0.4
        and required_acceleration >= (BOOST_ACCEL + 66.6666667) * max(delta, 13 / 120)
    )


def local_dodge_input(car: CarState, world_direction: Vec3) -> tuple[float, float]:
    local = car.local(world_direction.flat())
    if local.length() < 1e-6:
        return 0.0, 0.0
    forward_speed = car.velocity.dot(car.forward)
    speed_fraction = min(abs(forward_speed) / MAX_SPEED, 1.0)
    backwards = local.x < 0 if abs(forward_speed) < 100 else (local.x >= 0) == (forward_speed < 0)
    x = local.x / ((16 / 15) * (1 + 1.5 * speed_fraction) if backwards else 1)
    y = local.y / (1 + 0.9 * speed_fraction)
    return clamp(y / (abs(x) + abs(y))), clamp(-x / (abs(x) + abs(y)))


def turn_radius_at(speed: float) -> float:
    speed = clamp(speed, 0.0, MAX_SPEED)
    if speed <= 500:
        return 145 + (251 - 145) * speed / 500
    if speed <= 1000:
        return 251 + (425 - 251) * (speed - 500) / 500
    if speed <= 1500:
        return 425 + (727 - 425) * (speed - 1000) / 500
    if speed <= 1750:
        return 727 + (909 - 727) * (speed - 1500) / 250
    return 909 + (1136 - 909) * (speed - 1750) / 550


def speed_from_radius(radius: float) -> float:
    radius = clamp(radius, 145, 1136)
    if radius <= 251:
        return 500 * (radius - 145) / 106
    if radius <= 425:
        return 500 + 500 * (radius - 251) / 174
    if radius <= 727:
        return 1000 + 500 * (radius - 425) / 302
    if radius <= 909:
        return 1500 + 250 * (radius - 727) / 182
    return 1750 + 550 * (radius - 909) / 227
