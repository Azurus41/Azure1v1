"""Action states for movement, mechanics, ball control, and shots."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from rlbot.flat import ControllerState

from .constants import (
    CAR_AERIAL_OFFSET, CAR_FRONT_OFFSET, CAR_JUMP_OFFSET,
    JUMP_ACCEL, JUMP_VEL, MAX_SPEED, STICKY_ACCEL,
)
from .control import aerial_to, aim_at, drive_to, local_dodge_input, throttle_to, turn_radius_at
from .field import goal_target, in_field, nearest_surface, opponent_goal
from .model import BallSlice, CarState, World
from .vector import Vec3


class Action(Protocol):
    interruptible: bool
    finished: bool
    name: str

    def tick(self, world: World) -> ControllerState: ...


@dataclass(slots=True)
class DriveAction:
    target: Vec3
    target_speed: float = MAX_SPEED
    allow_boost: bool = True
    name: str = "drive"
    interruptible: bool = True
    finished: bool = False
    _distance: float = 100.0

    def tick(self, world: World) -> ControllerState:
        car = world.me
        surface = nearest_surface(self.target)
        projected = surface.project(self.target)
        self._distance = car.location.flat(surface.normal).flat_distance(projected, surface.normal)
        controls = ControllerState()
        drive_to(car, projected, controls, target_speed=self.target_speed, allow_boost=self.allow_boost, desired_up=surface.normal)
        if self._distance < 110 or self._distance > 35000:
            self.finished = True
        return controls


@dataclass(slots=True)
class BoostAction:
    target: Vec3
    require_full: bool = False
    name: str = "boost"
    interruptible: bool = True
    finished: bool = False

    def tick(self, world: World) -> ControllerState:
        controls = ControllerState()
        drive_to(world.me, self.target, controls, target_speed=MAX_SPEED, allow_boost=True)
        distance = world.me.location.flat_distance(self.target)
        self.finished = distance < 140 or world.me.boost > 95 or (self.require_full and world.me.boost > 90)
        return controls


@dataclass(slots=True)
class RecoveryAction:
    name: str = "recovery"
    interruptible: bool = False
    finished: bool = False
    _start: float = -1.0

    def tick(self, world: World) -> ControllerState:
        if self._start < 0:
            self._start = world.time
        controls = ControllerState()
        forward = world.me.velocity.flat() if world.me.velocity.length() > 500 else world.me.forward.flat()
        desired_up = Vec3(0, 0, 1) if world.me.up.z >= -0.1 else Vec3(0, 0, -1)
        aim_at(world.me, forward if forward.length() > 0.1 else world.me.forward, controls, desired_up)
        controls.throttle = 0.25
        self.finished = world.me.grounded or world.time - self._start > 1.35
        return controls


@dataclass(slots=True)
class DodgeAction:
    direction: Vec3
    jump_time: float = 0.1
    name: str = "dodge"
    interruptible: bool = False
    finished: bool = False
    _start: float = -1.0
    _jumping: bool = True
    _release_frames: int = 0
    _input: tuple[float, float] | None = None

    def tick(self, world: World) -> ControllerState:
        if self._start < 0:
            self._start = world.time
            self._jumping = world.me.grounded
        elapsed = world.time - self._start
        controls = ControllerState()
        if world.me.grounded and elapsed > (self.jump_time if self._jumping else 0) + 0.12:
            self.finished = True
        elif elapsed < self.jump_time and self._jumping:
            controls.jump = True
        elif self._release_frames < 3 and self._jumping:
            controls.jump = False
            self._release_frames += 1
        elif elapsed < (self.jump_time if self._jumping else 0) + 0.62:
            if self._input is None:
                self._input = local_dodge_input(world.me, self.direction)
            controls.yaw, controls.pitch = self._input
            controls.jump = True
        else:
            self.finished = True
        return controls


@dataclass(slots=True)
class HalfFlipAction:
    name: str = "half_flip"
    interruptible: bool = False
    finished: bool = False
    _start: float = -1.0
    _release: int = 0

    def tick(self, world: World) -> ControllerState:
        if self._start < 0:
            self._start = world.time
        elapsed = world.time - self._start
        controls = ControllerState()
        if elapsed < 0.1:
            controls.jump = world.me.grounded
        elif self._release < 3:
            self._release += 1
        elif elapsed < 0.32:
            controls.pitch = 1.0
            controls.jump = True
        elif elapsed < 1.15:
            aim_at(world.me, world.me.velocity.flat() or world.me.forward, controls, Vec3(0, 0, 1))
            controls.throttle = 0.6
        else:
            self.finished = True
        return controls


def _rotate_yaw(direction: Vec3, angle: float) -> Vec3:
    cosine, sine = math.cos(angle), math.sin(angle)
    return Vec3(
        direction.x * cosine - direction.y * sine,
        direction.x * sine + direction.y * cosine,
        0.0,
    )


@dataclass(slots=True)
class SpeedFlipAction:
    direction: Vec3
    name: str = "speed_flip"
    interruptible: bool = False
    finished: bool = False
    _start: float = -1.0
    _side: float = 1.0

    def tick(self, world: World) -> ControllerState:
        controls = ControllerState()
        controls.throttle = 1.0
        if self._start < 0:
            aim = self.direction.flat().normalized()
            forward_speed = max(world.me.velocity.dot(world.me.forward), 0.0)
            turn_angle = 0.06 * forward_speed / max(turn_radius_at(forward_speed), 1.0)
            left = _rotate_yaw(aim, turn_angle)
            right = _rotate_yaw(aim, -turn_angle)
            velocity = world.me.velocity.flat()
            if velocity.length() < 200:
                self.finished = True
                return controls
            if velocity.angle(left) < velocity.angle(right):
                pre_turn, self._side = left, 1.0
            else:
                pre_turn, self._side = right, -1.0
            aim_at(world.me, pre_turn, controls)
            if velocity.flat_angle(pre_turn) < 0.05:
                self._start = world.time
            if not world.me.grounded or not in_field(world.me.location, 150):
                self.finished = True
            return controls
        elapsed = world.time - self._start
        if 0 < elapsed < 0.1:
            controls.jump = True
        elif 0.12 < elapsed < 0.15:
            controls.pitch = -1.0
            controls.roll = self._side * 0.5
            controls.jump = True
        elif 0.15 < elapsed < 0.75:
            controls.pitch = 1.0
            controls.roll = self._side
        elif 0.75 < elapsed < 0.9:
            controls.pitch = 1.0
            controls.handbrake = True
            controls.yaw = self._side
        else:
            self.finished = True
        if not in_field(world.me.location, 150):
            self.finished = True
        return controls


@dataclass(slots=True)
class KickoffAction:
    name: str = "kickoff"
    interruptible: bool = False
    finished: bool = False
    _speed_flip: SpeedFlipAction | None = None
    _final_dodge: DodgeAction | None = None
    _time_on_ground: float = 0.0
    _diagonal: bool | None = None
    _speed_flipped: bool = False
    _initial_touch: float | None = None

    def tick(self, world: World) -> ControllerState:
        controls = ControllerState()
        if self._initial_touch is None:
            self._initial_touch = world.latest_touch_time
        if world.latest_touch_time > self._initial_touch + 1e-6:
            self.finished = True
            return controls
        if not world.kickoff:
            self.finished = True
            return controls
        if self._final_dodge is not None and not self._final_dodge.finished:
            return self._final_dodge.tick(world)
        if self._speed_flip is not None and not self._speed_flip.finished:
            controls = self._speed_flip.tick(world)
            controls.boost = world.me.boost > 0
            return controls
        if self._speed_flipped and world.me.grounded:
            self._time_on_ground += world.delta

        car = world.me
        if self._diagonal is None and car.velocity.length() < 200:
            self._diagonal = abs(car.location.x) > 1000
        diagonal = bool(self._diagonal)
        shot_direction = (opponent_goal(car.team) - world.ball.location).flat().normalized()
        if not shot_direction.length():
            shot_direction = Vec3(0, 1 if car.team == 0 else -1, 0)
        throttle_to(car, MAX_SPEED, controls, allow_boost=False)

        if not diagonal or self._speed_flipped:
            offset = 170.0 if self._speed_flipped else 2600.0
            aim = world.ball.location - shot_direction * offset
            aim_at(car, aim - car.location, controls)
            if not diagonal and not self._speed_flipped:
                controls.steer *= 0.4
        elif car.velocity.length() > 500:
            aim_at(car, world.ball.location - car.location, controls)

        speed_threshold = 600.0 if diagonal else 700.0 + abs(car.location.x) * 3.0
        if not self._speed_flipped and car.velocity.length() > speed_threshold:
            target = world.ball.location - shot_direction * (250.0 if diagonal else -1000.0)
            direction = (target - car.location).flat().normalized()
            if direction.length() > 0.1:
                self._speed_flipped = True
                self._speed_flip = SpeedFlipAction(direction)
        elif (
            self._speed_flipped
            and car.location.flat_distance(world.ball.location) < 800
            and self._time_on_ground > 0.1
        ):
            self._final_dodge = DodgeAction(shot_direction, jump_time=0.18)
            return self._final_dodge.tick(world)
        return controls


@dataclass(slots=True)
class WaveDashAction:
    direction: Vec3
    jump_time: float = 0.05
    name: str = "wavedash"
    interruptible: bool = False
    finished: bool = False
    _start: float = -1.0
    _jumping: bool = True

    def tick(self, world: World) -> ControllerState:
        if self._start < 0:
            self._start = world.time
            self._jumping = world.me.grounded
        elapsed = world.time - self._start
        controls = ControllerState()
        if elapsed < self.jump_time and self._jumping:
            controls.jump = True
        elif not world.me.grounded and world.me.location.z < 45 and world.me.velocity.z < -80:
            controls.yaw, controls.pitch = local_dodge_input(world.me, self.direction)
            controls.jump = True
        elif not world.me.grounded:
            surface = nearest_surface(world.me.location)
            controls.throttle = 0.25
            aim_at(world.me, self.direction.flat(surface.normal) + surface.normal * 0.2, controls, surface.normal)
        else:
            self.finished = True
        return controls


@dataclass(slots=True)
class DribbleAction:
    target: Vec3
    name: str = "dribble"
    interruptible: bool = True
    finished: bool = False
    _start: float = -1.0

    def tick(self, world: World) -> ControllerState:
        if self._start < 0:
            self._start = world.time
        surface = nearest_surface(world.ball.location)
        attack = (goal_target(world.me.team) - world.ball.location).flat(surface.normal).normalized()
        if attack.length() < 0.1:
            attack = Vec3(0, 1 if world.me.team == 0 else -1, 0)
        contact = world.ball.location - attack * 185
        contact = surface.project(contact)
        controls = ControllerState()
        drive_to(world.me, contact, controls, target_speed=1650, allow_boost=world.me.boost > 55, desired_up=surface.normal)
        if not world.me.grounded or world.time - self._start > 2.8:
            self.finished = True
        return controls



@dataclass(slots=True)
class GroundShotAction:
    intercept_time: float
    ball_location: Vec3
    shot_target: Vec3
    name: str = "ground_shot"
    interruptible: bool = True
    finished: bool = False
    contact_location: Vec3 = Vec3()
    shot_direction: Vec3 = Vec3(0, 1, 0)
    _last_retarget: float = -10.0
    _initial_touch: float = -1.0

    def __post_init__(self):
        self._update_geometry(self.ball_location, self.shot_target)

    def tick(self, world: World) -> ControllerState:
        if self._initial_touch < 0:
            self._initial_touch = world.latest_touch_time
        if self.interruptible and world.time - self._last_retarget >= 0.2:
            current = nearest_slice(world.ball.slices, self.intercept_time)
            if current is not None:
                self._last_retarget = world.time
                self._update_geometry(current.location, self.shot_target)
        controls = ControllerState()
        if self.interruptible:
            surface = nearest_surface(self.contact_location)
            target = surface.project(self.contact_location)
            drive_to(world.me, target, controls, target_speed=MAX_SPEED, allow_boost=True, desired_up=surface.normal)
            eta = estimate_local_eta(world.me, target)
            remaining = self.intercept_time - world.time
            if world.latest_touch_time > self._initial_touch:
                self.finished = True
            elif remaining < -0.12 or eta > remaining + 0.1:
                self.finished = True
            elif world.me.location.flat_distance(world.ball.location) < 145 and remaining < 0.28:
                self.finished = True
        return controls

    def _update_geometry(self, ball_location: Vec3, target: Vec3) -> None:
        surface = nearest_surface(ball_location)
        self.shot_direction = (target - ball_location).flat(surface.normal).normalized()
        if self.shot_direction.length() < 0.1:
            self.shot_direction = (target - ball_location).flat(surface.normal).normalized() or Vec3(0, 1, 0)
        self.contact_location = ball_location - self.shot_direction * CAR_FRONT_OFFSET


@dataclass(slots=True)
class JumpShotAction:
    intercept_time: float
    ball_location: Vec3
    shot_target: Vec3
    name: str = "jump_shot"
    interruptible: bool = True
    finished: bool = False
    contact_location: Vec3 = Vec3()
    shot_direction: Vec3 = Vec3(0, 1, 0)
    _jumped: bool = False
    _jump_elapsed: float = 0.0
    _release_frames: int = 0
    _last_retarget: float = -10.0
    _initial_touch: float = -1.0

    def __post_init__(self):
        self._update_geometry(self.ball_location, self.shot_target)

    def tick(self, world: World) -> ControllerState:
        if self._initial_touch < 0:
            self._initial_touch = world.latest_touch_time
        remaining = self.intercept_time - world.time
        if self.interruptible and world.time - self._last_retarget >= 0.2:
            current = nearest_slice(world.ball.slices, self.intercept_time)
            if current is not None:
                self._last_retarget = world.time
                self._update_geometry(current.location, self.shot_target)
        controls = ControllerState()
        if not self._jumped:
            surface = nearest_surface(self.contact_location)
            target = surface.project(self.contact_location)
            drive_to(world.me, target, controls, target_speed=MAX_SPEED, allow_boost=True, desired_up=surface.normal)
            height = max(20.0, surface.height_above(self.contact_location) - 17)
            jump_time = time_to_jump(height)
            eta = estimate_local_eta(world.me, target)
            if remaining < -0.1 or eta > remaining + 0.12:
                self.finished = True
            elif remaining < jump_time:
                predicted_jump = _location_after_jump(world.me, remaining, world.gravity)
                tolerance = 55.0 - surface.normal.dot(Vec3(0, 0, 1)) * 20.0
                if predicted_jump.flat_distance(target, surface.normal) < tolerance:
                    self._jumped = True
                    self.interruptible = False
        else:
            self._jump_elapsed += world.delta
            surface = nearest_surface(self.contact_location)
            dodge_direction = self.shot_direction.flat(surface.normal).normalized()
            aim_at(world.me, dodge_direction, controls, surface.normal)
            if remaining < -0.12 or world.latest_touch_time > self._initial_touch:
                self.finished = True
            elif remaining > 0.075 or self._jump_elapsed < 0.05:
                controls.jump = True
            elif self._release_frames < 3:
                self._release_frames += 1
            elif remaining > 0.02:
                pass
            else:
                controls.yaw, controls.pitch = local_dodge_input(world.me, self.shot_direction)
                controls.jump = True
                if self._jump_elapsed > 0.5:
                    self.finished = True
        return controls

    def _update_geometry(self, ball_location: Vec3, target: Vec3) -> None:
        self.shot_direction = (target - ball_location).normalized() or Vec3(0, 1, 0)
        self.contact_location = ball_location - self.shot_direction * CAR_JUMP_OFFSET


@dataclass(slots=True)
class DoubleJumpShotAction(JumpShotAction):
    name: str = "double_jump_shot"
    _double_release: int = 0
    _double_applied: bool = False

    def tick(self, world: World) -> ControllerState:
        if self._initial_touch < 0:
            self._initial_touch = world.latest_touch_time
        remaining = self.intercept_time - world.time
        if self.interruptible and world.time - self._last_retarget >= 0.2:
            current = nearest_slice(world.ball.slices, self.intercept_time)
            if current is not None:
                self._last_retarget = world.time
                self._update_geometry(current.location, self.shot_target)
        controls = ControllerState()
        if not self._jumped:
            surface = nearest_surface(self.contact_location)
            target = surface.project(self.contact_location)
            drive_to(world.me, target, controls, target_speed=MAX_SPEED, allow_boost=True, desired_up=surface.normal)
            height = max(20.0, surface.height_above(self.contact_location) - 17)
            if remaining < time_to_jump(height):
                predicted_jump = _location_after_jump(world.me, remaining, world.gravity)
                if predicted_jump.flat_distance(target, surface.normal) < 55.0:
                    self._jumped = True
                    self.interruptible = False
        else:
            self._jump_elapsed += world.delta
            surface = nearest_surface(self.contact_location)
            if self._jump_elapsed <= 0.2:
                aim_at(world.me, self.contact_location - world.me.location, controls, surface.normal)
                controls.jump = True
            elif self._double_release < 3:
                controls.jump = False
                self._double_release += 1
            elif self._jump_elapsed <= 0.275:
                controls.jump = True
                controls.pitch = controls.yaw = controls.roll = controls.steer = 0.0
            else:
                if not self._double_applied:
                    self._double_applied = True
                    self._jump_elapsed = 0.0
                desired_up = (self.contact_location - world.me.location).normalized()
                if desired_up.length() < 0.1:
                    desired_up = self.shot_direction
                aerial_to(
                    world.me, self.contact_location, max(remaining, 0.01), controls,
                    desired_up=desired_up, boost=True, delta=world.delta,
                )
            if remaining < -0.1 or world.latest_touch_time > self._initial_touch:
                self.finished = True
        return controls



@dataclass(slots=True)
class AerialShotAction:
    intercept_time: float
    ball_location: Vec3
    shot_target: Vec3
    double_jump: bool
    name: str = "aerial"
    interruptible: bool = True
    finished: bool = False
    contact_location: Vec3 = Vec3()
    shot_direction: Vec3 = Vec3(0, 1, 0)
    _aerialing: bool = False
    _jumped: bool = False
    _jump_elapsed: float = 0.0
    _release: int = 0
    _initial_touch: float = -1.0
    _start_boost: float = 0.0
    _last_retarget: float = -10.0

    def __post_init__(self):
        self._update_geometry(self.ball_location, self.shot_target)

    def tick(self, world: World) -> ControllerState:
        if self._initial_touch < 0:
            self._initial_touch = world.latest_touch_time
            self._start_boost = world.me.boost
        remaining = self.intercept_time - world.time
        controls = ControllerState()
        if not self._aerialing:
            current = nearest_slice(world.ball.slices, self.intercept_time)
            if world.time - self._last_retarget >= 0.2 and current is not None:
                self._last_retarget = world.time
                self._update_geometry(current.location, self.shot_target)
            surface = nearest_surface(world.me.location)
            target = self.contact_location.flat(surface.normal)
            target = surface.project(target)
            drive_to(world.me, target, controls, target_speed=MAX_SPEED, allow_boost=True, desired_up=surface.normal)
            to_contact = (self.contact_location - world.me.location).flat()
            angle = math.acos(max(-1.0, min(1.0, world.me.forward.flat().normalized().dot(to_contact.normalized())))) if to_contact.length() > 1 else math.pi
            if remaining <= 0.45 or (angle < 0.25 and remaining < 1.35):
                self._aerialing = True
                self.interruptible = False
        if self._aerialing:
            self._jump_elapsed += world.delta
            surface = nearest_surface(world.me.location)
            if not self._jumped:
                aim_at(world.me, self.contact_location - world.me.location, controls, surface.normal)
                controls.jump = self._jump_elapsed <= 0.2
                if self._jump_elapsed > 0.2 and not self.double_jump:
                    self._jumped = True
                elif self._jump_elapsed > 0.205 and self.double_jump and self._release < 3:
                    controls.jump = False
                    self._release += 1
                elif self._jump_elapsed > 0.225 and self.double_jump:
                    controls.jump = True
                    if self._jump_elapsed > 0.25:
                        self._jumped = True
            else:
                desired_up = (self.contact_location - world.me.location).normalized()
                if desired_up.length() < 0.1:
                    desired_up = self.shot_direction
                aerial_to(
                    world.me, self.contact_location, max(remaining, 0.01), controls,
                    desired_up=desired_up, boost=True, delta=world.delta,
                )
            if world.latest_touch_time > self._initial_touch or remaining < -0.08:
                self.finished = True
        elif world.me.boost <= 0 or world.me.boost > self._start_boost + 1.0:
            self.finished = True
        return controls

    def _update_geometry(self, ball_location: Vec3, target: Vec3) -> None:
        self.shot_direction = (target - ball_location).normalized() or Vec3(0, 1, 0)
        self.contact_location = ball_location - self.shot_direction * CAR_AERIAL_OFFSET


def nearest_slice(slices: tuple[BallSlice, ...], time: float) -> BallSlice | None:
    return min(slices, key=lambda item: abs(item.time - time), default=None)


def estimate_local_eta(car: CarState, target: Vec3) -> float:
    distance = car.location.flat_distance(target)
    speed = max(car.velocity.dot(car.forward), 0.0)
    average = max(700.0, (speed + MAX_SPEED) * 0.5)
    turn = car.forward.flat().angle((target - car.location).flat()) * 0.08
    return turn + distance / average


def _location_after_jump(car: CarState, seconds: float, gravity: float) -> Vec3:
    jump_time = min(max(seconds, 0.0), 0.2)
    sticky_time = min(max(seconds, 0.0), 0.05)
    return (
        car.location
        + car.velocity * seconds
        + Vec3(0, 0, 0.5 * gravity * seconds * seconds)
        + car.up * JUMP_VEL * seconds
        + car.up * JUMP_ACCEL * jump_time * max(seconds - 0.5 * jump_time, 0.0)
        - car.up * STICKY_ACCEL * sticky_time * max(seconds - 0.5 * sticky_time, 0.0)
    )


def time_to_jump(height: float) -> float:
    if height <= 0:
        return 0.05
    # Party's first-jump apex is roughly 0.64s for a full hold.
    return min(0.64, 0.05 + math.sqrt(max(0.0, 2.0 * max(0.0, height - 11.66) / 830.0)))
