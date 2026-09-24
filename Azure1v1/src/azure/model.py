"""Immutable world snapshots used by strategy and action code."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from .constants import DEFAULT_GRAVITY
from .field import BoostPad
from .vector import Vec3, vec


@dataclass(frozen=True, slots=True)
class CarState:
    index: int
    team: int
    location: Vec3
    velocity: Vec3
    angular_velocity: Vec3
    forward: Vec3
    right: Vec3
    up: Vec3
    boost: float
    grounded: bool
    jumped: bool
    double_jumped: bool
    demolished: bool
    latest_touch_time: float = -1.0
    latest_touch_index: int = -1

    def local(self, value: Vec3) -> Vec3:
        return Vec3(value.dot(self.forward), value.dot(self.right), value.dot(self.up))

    def predict(self, seconds: float, gravity: float) -> tuple[Vec3, Vec3]:
        acceleration = Vec3(0, 0, gravity)
        location = self.location + self.velocity * seconds + acceleration * (0.5 * seconds * seconds)
        return location, self.velocity + acceleration * seconds

    def landing_time(self, gravity: float) -> float:
        if self.grounded:
            return 0.0
        roots = quadratic_roots(0.5 * gravity, self.velocity.z, self.location.z - 17.0)
        valid = [root for root in roots if root > 0]
        return min(valid) if valid else 0.0


@dataclass(frozen=True, slots=True)
class BallSlice:
    time: float
    location: Vec3
    velocity: Vec3

    def predict(self, seconds: float, gravity: float) -> Vec3:
        return self.location + self.velocity * seconds + Vec3(0, 0, 0.5 * gravity * seconds * seconds)


@dataclass(frozen=True, slots=True)
class BallState:
    location: Vec3
    velocity: Vec3
    slices: tuple[BallSlice, ...] = ()


@dataclass(frozen=True, slots=True)
class World:
    time: float
    delta: float
    gravity: float
    phase: Any
    ball: BallState
    me: CarState
    teammates: tuple[CarState, ...]
    opponents: tuple[CarState, ...]
    boost_pads: tuple[BoostPad, ...]
    kickoff: bool
    demolition: bool
    latest_touch_time: float
    latest_touch_index: int

    @property
    def living_opponents(self) -> tuple[CarState, ...]:
        return tuple(car for car in self.opponents if not car.demolished)


@dataclass(frozen=True, slots=True)
class BoostTracker:
    pads: tuple[BoostPad, ...]



def snapshot(
    packet: Any, *, player_id: int, team: int, previous_time: float,
    tracker: BoostTracker | None, prediction: Any,
) -> tuple[World, BoostTracker]:
    time = float(packet.match_info.seconds_elapsed)
    raw_delta = time - previous_time
    delta = min(max(raw_delta if raw_delta > 0 else 1 / 120, 1 / 240), 1 / 30)
    gravity = float(getattr(packet.match_info, "world_gravity_z", DEFAULT_GRAVITY))
    phase = packet.match_info.match_phase
    kickoff = _phase_name(phase) == "MatchPhase.Kickoff"
    if tracker is None:
        raise ValueError("BoostTracker must be initialized with field_info")
    state_by_index = {index: state for index, state in enumerate(packet.boost_pads)}
    pads = tuple(
        BoostPad(
            pad.index, pad.location, pad.is_large,
            bool(getattr(state_by_index.get(pad.index), "is_active", True)),
            float(getattr(state_by_index.get(pad.index), "timer", 0.0)),
        )
        for pad in tracker.pads
    )
    cars = [_car_from_player(index, player) for index, player in enumerate(packet.players)]
    me = next((car for car in cars if car.index == player_id), None)
    if me is None:
        me = _car_from_player(0, packet.players[0])
    ball_info = packet.balls[0] if packet.balls else None
    raw_slices = getattr(prediction, "slices", ()) if prediction else ()
    slices = tuple(
        BallSlice(float(item.game_seconds), vec(item.physics.location), vec(item.physics.velocity))
        for item in raw_slices
    )
    ball = BallState(
        vec(ball_info.physics.location) if ball_info else Vec3(),
        vec(ball_info.physics.velocity) if ball_info else Vec3(), slices,
    )
    touch_time, touch_index = _latest_touch(cars)
    world = World(
        time=time, delta=delta, gravity=gravity, phase=phase, ball=ball, me=me,
        teammates=tuple(car for car in cars if car.team == team and car.index != me.index),
        opponents=tuple(car for car in cars if car.team != team), boost_pads=pads,
        kickoff=kickoff, demolition=me.demolished,
        latest_touch_time=touch_time, latest_touch_index=touch_index,
    )
    return world, tracker


def quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]:
    discriminant = b * b - 4.0 * a * c
    if a == 0 or discriminant < 0:
        return -1.0, -1.0
    root = math.sqrt(discriminant)
    return (-b + root) / (2 * a), (-b - root) / (2 * a)


def _car_from_player(index: int, player: Any) -> CarState:
    rotation = player.physics.rotation
    pitch, yaw, roll = float(rotation.pitch), float(rotation.yaw), float(rotation.roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = Vec3(cp * cy, cp * sy, sp)
    right = Vec3(cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr)
    up = Vec3(-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr)
    air_state = str(player.air_state)
    touch = player.latest_touch
    return CarState(
        index=int(player.player_id), team=int(player.team),
        location=vec(player.physics.location), velocity=vec(player.physics.velocity),
        angular_velocity=vec(player.physics.angular_velocity),
        forward=forward, right=right, up=up, boost=float(player.boost),
        grounded=air_state == "AirState.OnGround",
        jumped=air_state == "AirState.Jumping",
        double_jumped=air_state in ("AirState.DoubleJumping", "AirState.Dodging"),
        demolished=float(player.demolished_timeout) > 0.0,
        latest_touch_time=float(touch.game_seconds) if touch else -1.0,
        latest_touch_index=int(player.player_id),
    )


def _latest_touch(cars: Iterable[CarState]) -> tuple[float, int]:
    touches = [(car.latest_touch_time, car.latest_touch_index) for car in cars if car.latest_touch_time >= 0]
    return max(touches, default=(-1.0, -1))


def _phase_name(phase: Any) -> str:
    return str(phase)
