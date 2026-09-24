"""Strategy layer: predict, prioritize, and select Party-style actions."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .constants import (
    BOOST_ACCEL, BOOST_CONSUMPTION, JUMP_ACCEL, JUMP_VEL, BALL_RADIUS,
    MAX_SPEED, MAX_THROTTLE_SPEED, STICKY_ACCEL, THROTTLE_ACCEL,
)
from .actions import (
    Action, AerialShotAction, BoostAction, DoubleJumpShotAction,
    DribbleAction, DriveAction, GroundShotAction, JumpShotAction,
    RecoveryAction,
)
from .field import BoostPad, clear_target, goal_target, nearest_surface, own_goal, point_is_in_goal
from .model import BallSlice, CarState, World
from .vector import Vec3


def estimate_eta(car: CarState, target: Vec3) -> float:
    """Fast, conservative intercept ETA used for saves and opponent races."""
    distance = car.location.flat_distance(target)
    if distance < 100:
        return 0.0
    direction = (target - car.location).flat().normalized()
    forward = car.forward.flat().normalized()
    speed = max(car.velocity.dot(forward), 0.0)
    angle = math.acos(max(-1.0, min(1.0, forward.dot(direction))))
    turn_time = 0.0 if angle < 0.2 else angle * (145 + 0.43 * speed) / max(1200.0, speed)
    remaining = max(0.0, distance - 180)
    boost_time = min(car.boost / BOOST_CONSUMPTION, 2.4) if car.boost > 0 and speed > 400 else 0.0
    speed_after = min(speed + BOOST_ACCEL * boost_time, MAX_SPEED)
    boost_distance = 0.5 * (speed + speed_after) * boost_time
    if boost_distance >= remaining:
        acceleration_time = 0.0
        if BOOST_ACCEL > 0:
            acceleration_time = (math.sqrt(max(0.0, speed * speed + 2 * BOOST_ACCEL * remaining)) - speed) / BOOST_ACCEL
        return (turn_time + acceleration_time) * 1.04
    time = turn_time + boost_time
    remaining -= boost_distance
    speed = speed_after
    if speed < MAX_THROTTLE_SPEED:
        throttle_time = (MAX_THROTTLE_SPEED - speed) / THROTTLE_ACCEL
        throttle_distance = speed * throttle_time + 0.5 * THROTTLE_ACCEL * throttle_time * throttle_time
        if throttle_distance >= remaining:
            return (time + max(0.0, 2 * remaining / max(speed + MAX_THROTTLE_SPEED, 1))) * 1.04
        time += throttle_time
        remaining -= throttle_distance
        speed = MAX_THROTTLE_SPEED
    return (time + remaining / max(speed, 700.0)) * 1.04


def opponent_reach(car: CarState, target: Vec3) -> float:
    return estimate_eta(car, target)


def find_best_boost(car: CarState, pads: tuple, teammates: tuple[CarState, ...] = ()) -> BoostPad | None:
    candidates = []
    for pad in pads:
        if not pad.is_large or (not pad.is_active and pad.timer > estimate_eta(car, pad.location) + 0.15):
            continue
        my_eta = estimate_eta(car, pad.location)
        reserved = any(estimate_eta(mate, pad.location) + 0.08 < my_eta for mate in teammates)
        if not reserved:
            candidates.append((my_eta, pad))
    return min(candidates, default=(0.0, None), key=lambda item: item[0])[1]


@dataclass(frozen=True, slots=True)
class Candidate:
    action: Action
    score: float
    intercept_time: float


def choose_action(world: World) -> Action | None:
    if world.demolition:
        return None
    if world.kickoff and (
        world.latest_touch_time < 0
        or world.time - world.latest_touch_time > 0.5
    ):
        from .actions import KickoffAction
        return KickoffAction()
    if not world.ball.slices:
        return None
    if not world.me.grounded:
        if world.me.up.z < 0.25 or not world.me.jumped:
            return RecoveryAction()
        return DriveAction(world.ball.location, target_speed=1900, allow_boost=world.me.boost > 20)

    save = find_save(world)
    if save is not None:
        return save
    shot = find_shot(world)
    if shot is not None:
        target = min(
            world.ball.slices,
            key=lambda item: abs(item.time - shot.intercept_time),
        )
        if not opponents_win_race(world, target.location, shot.intercept_time):
            return shot.action

    distance = world.me.location.flat_distance(world.ball.location)
    if distance < 420 and world.ball.location.z < 130:
        surface = nearest_surface(world.ball.location)
        if world.me.up.dot(surface.normal) > 0.9 and world.ball.velocity.dot(surface.normal) > -250:
            return DribbleAction(world.ball.location)

    if world.me.boost < 28:
        pad = find_best_boost(world.me, world.boost_pads, world.teammates)
        if pad is not None:
            return BoostAction(pad.location, require_full=True)

    if opponents_win_race(world, world.ball.location, estimate_eta(world.me, world.ball.location) + 0.35):
        target = clear_target(world.ball.location, world.me.team)
        return DriveAction(target, target_speed=1500, allow_boost=False, name="defensive_clear")
    return DriveAction(world.ball.location, target_speed=MAX_SPEED, allow_boost=True, name="ball_chase")


def find_save(world: World) -> Action | None:
    own = own_goal(world.me.team)
    attack_toward_own = (own - world.ball.location).flat().normalized().dot(world.ball.velocity.flat().normalized()) > 0.35
    for item in _sample_slices(world.ball.slices):
        remaining = item.time - world.time
        if remaining < 0.08 or remaining > 3.2:
            continue
        threat = point_is_in_goal(item.location, world.me.team)
        imminent = attack_toward_own and item.location.flat_distance(own) < 3000 and remaining < 2.0
        if not threat and not imminent:
            continue
        if estimate_eta(world.me, item.location) > remaining + 0.08:
            continue
        target = clear_target(item.location, world.me.team)
        if item.location.z < 105:
            target = Vec3(target.x, target.y, 80)
        if item.location.z > 320:
            jump_mode = aerial_jump_mode(world.me, item.location, remaining, world.gravity)
            if jump_mode >= 0:
                return AerialShotAction(item.time, item.location, target, bool(jump_mode), name="save_aerial")
        if item.location.z > 175:
            return DoubleJumpShotAction(item.time, item.location, target, name="save_double_jump")
        if item.location.z > 105:
            return JumpShotAction(item.time, item.location, target, name="save_jump")
        return GroundShotAction(item.time, item.location, target, name="save_ground")
    return None


def find_shot(world: World) -> Candidate | None:
    best: Candidate | None = None
    for item in _sample_slices(world.ball.slices):
        remaining = item.time - world.time
        if remaining < 0.16 or remaining > 3.0:
            continue
        contact = item.location - (goal_target(world.me.team) - item.location).normalized() * 120
        eta = estimate_eta(world.me, contact)
        if eta > remaining + 0.08:
            continue
        surface = nearest_surface(item.location)
        ball_height = surface.height_above(item.location)
        target = goal_target(world.me.team, away=item.time - world.latest_touch_time > 3.0 and world.me.boost < 20)
        action: Action | None = None
        bonus = 0.0
        if ball_height <= BALL_RADIUS + 30 and item.location.z < 230:
            action = GroundShotAction(item.time, item.location, target, name="power_shot")
        elif ball_height < 280 and item.location.z < 500:
            action = JumpShotAction(item.time, item.location, target, name="jump_shot")
        elif item.location.z < 610:
            action = DoubleJumpShotAction(item.time, item.location, target, name="double_jump_shot")
        if item.location.z > 310:
            jump_mode = aerial_jump_mode(world.me, item.location, remaining, world.gravity)
            if jump_mode >= 0:
                action = AerialShotAction(item.time, item.location, target, bool(jump_mode), name="aerial")
                bonus = 0.12
        if action is None:
            continue
        score = remaining - bonus - max(0.0, eta - remaining + 0.2) * 2.0
        if best is None or score < best.score:
            best = Candidate(action, score, item.time)
    return best



def _sample_slices(slices: tuple[BallSlice, ...]):
    near_count = min(len(slices), 240)
    yield from (slices[index] for index in range(0, near_count, 2))
    yield from (slices[index] for index in range(near_count, len(slices), 5))


def opponents_win_race(world: World, target: Vec3, our_eta: float) -> bool:
    return any(opponent_reach(car, target) < our_eta - 0.10 for car in world.living_opponents)


def aerial_jump_mode(
    car: CarState, ball_location: Vec3, seconds: float,
    gravity: float = -650.0,
) -> int:
    if seconds <= 0.12 or ball_location.z < 280:
        return -1
    shot_direction = (goal_target(car.team) - ball_location).normalized()
    contact = ball_location - shot_direction * 155.0
    single = _aerial_boost_estimate(car, contact, seconds, False, gravity)
    double = _aerial_boost_estimate(car, contact, seconds, True, gravity)
    if double >= 0 and (single < 0 or double < single):
        return 1
    return 0 if single >= 0 else -1


def aerial_possible(
    car: CarState, ball_location: Vec3, seconds: float,
    double_jump: bool, gravity: float = -650.0,
) -> bool:
    mode = aerial_jump_mode(car, ball_location, seconds, gravity)
    return mode == (1 if double_jump else 0)


def _aerial_boost_estimate(
    car: CarState, target: Vec3, seconds: float,
    double_jump: bool, gravity: float,
) -> float:
    acceleration = Vec3(0, 0, gravity)
    jump_time = min(seconds, 0.2) if car.grounded else 0.0
    sticky_time = min(seconds, 0.05) if car.grounded else 0.0
    final_position = car.location + car.velocity * seconds + acceleration * (0.5 * seconds * seconds)
    final_velocity = car.velocity + acceleration * seconds
    if car.grounded:
        final_position += car.up * (
            JUMP_VEL * seconds
            + JUMP_ACCEL * jump_time * (seconds - 0.5 * jump_time)
            - STICKY_ACCEL * sticky_time * (seconds - 0.5 * sticky_time)
        )
        if double_jump:
            final_position += car.up * JUMP_VEL * max(seconds - jump_time, 0.0)
        final_velocity += car.up * (
            JUMP_VEL * (2.0 if double_jump else 1.0)
            + JUMP_ACCEL * jump_time
            - STICKY_ACCEL * sticky_time
        )
    offset = target - final_position
    direction = offset.normalized()
    if direction.length() < 0.1:
        return -1.0
    angle = max(0.0001, direction.angle(car.forward))
    turn_time = 0.6 * (2.0 * math.sqrt(angle / 9.0))
    tau_one = turn_time * max(0.0, min(1.0, 1.0 - 0.4 / angle))
    travel_time = max(0.01, seconds - tau_one)
    required_acceleration = 2.0 * offset.length() / (travel_time * travel_time)
    total_acceleration = BOOST_ACCEL + 66.6666667
    ratio = required_acceleration / total_acceleration
    tau_two = seconds - travel_time * math.sqrt(max(0.0, 1.0 - max(0.0, min(1.0, ratio))))
    estimated_velocity = final_velocity + total_acceleration * max(0.0, tau_two - tau_one) * direction
    boost_required = max(0.0, tau_two - tau_one) * BOOST_CONSUMPTION
    enough_boost = boost_required < car.boost * 0.9
    enough_time = abs(ratio) < 0.9
    return boost_required if estimated_velocity.length() < MAX_SPEED * 0.9 and enough_boost and enough_time else -1.0
