"""Pure deterministic tests for Azure's physics and decision helpers."""

import sys
import unittest
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlbot.flat import (
    AirState, BallInfo, BallPrediction, BoostPad as FlatBoostPad, BoostPadState, ControllerState,
    GamePacket, MatchInfo, MatchPhase, Physics, PlayerInfo, PredictionSlice,
    Rotator, Touch, Vector3,
)

from azure.actions import DoubleJumpShotAction, JumpShotAction, KickoffAction, SpeedFlipAction
from azure.bot import AzureBot
from azure.constants import MAX_SPEED
from azure.control import turn_radius_at
from azure.field import BoostPad, clamp_target, goal_target, point_is_in_goal
from azure.model import BallSlice, BallState, BoostTracker, CarState, World, snapshot
from azure.strategy import aerial_possible, choose_action, estimate_eta, find_best_boost, opponent_reach
from azure.vector import Vec3


class VectorTests(unittest.TestCase):
    def test_vector_operations(self):
        a = Vec3(3, 4, 0)
        self.assertEqual(a.length(), 5.0)
        self.assertEqual(a.normalized().length(), 1.0)
        self.assertAlmostEqual(a.cross(Vec3(0, 0, 1)).dot(Vec3(0, 0, 1)), 0.0)

    def test_team_direction_is_symmetric(self):
        blue = goal_target(0)
        orange = goal_target(1)
        self.assertGreater(blue.y, 5000)
        self.assertLess(orange.y, -5000)
        self.assertTrue(point_is_in_goal(Vec3(0, -5200, 320), 0))
        self.assertTrue(point_is_in_goal(Vec3(0, 5200, 320), 1))


class ControlTests(unittest.TestCase):
    def test_turn_radius_is_monotonic(self):
        values = [turn_radius_at(speed) for speed in (0, 500, 1000, 1500, 1750, MAX_SPEED)]
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[-1], values[0])


class TargetTests(unittest.TestCase):
    def test_clamp_target_respects_posts_and_crossbar(self):
        target = clamp_target(Vec3(5000, 4000, 5000), 0)
        self.assertLess(abs(target.x), 890)
        self.assertLess(target.z, 640)
        self.assertGreater(target.y, 5120)


class PacketAdapterTests(unittest.TestCase):
    def test_real_flatbuffer_packet_snapshot(self):
        zero = Vector3()
        blue_physics = Physics(
            location=Vector3(0, -3000, 17), rotation=Rotator(0, 0, 0),
            velocity=Vector3(500, 0, 0), angular_velocity=zero,
        )
        orange_physics = Physics(
            location=Vector3(500, 3000, 17), rotation=Rotator(0, 3.14159265, 0),
            velocity=Vector3(-500, 0, 0), angular_velocity=zero,
        )
        players = [
            PlayerInfo(
                physics=blue_physics, air_state=AirState.OnGround,
                latest_touch=Touch(game_seconds=1.25), team=0, boost=33, player_id=0,
            ),
            PlayerInfo(
                physics=orange_physics, air_state=AirState.OnGround,
                latest_touch=Touch(game_seconds=2.5), team=1, boost=75, player_id=1,
            ),
        ]
        packet = GamePacket(
            players=players,
            boost_pads=[BoostPadState(is_active=False, timer=4.0)],
            balls=[BallInfo(physics=Physics(location=Vector3(0, 0, 93), velocity=Vector3(0, 1000, 0), angular_velocity=zero))],
            match_info=MatchInfo(seconds_elapsed=3.0, match_phase=MatchPhase.Active, world_gravity_z=-650),
            teams=[],
        )
        prediction = BallPrediction(slices=[
            PredictionSlice(game_seconds=3.0, physics=packet.balls[0].physics),
            PredictionSlice(game_seconds=3.5, physics=Physics(location=Vector3(0, 500, 93), velocity=Vector3(0, 1000, 0), angular_velocity=zero)),
        ])
        tracker = BoostTracker((BoostPad(0, Vec3(1000, 0, 0), True),))
        world, _ = snapshot(
            packet, player_id=0, team=0, previous_time=2.9,
            tracker=tracker, prediction=prediction,
        )
        self.assertEqual(world.me.index, 0)
        self.assertTrue(world.me.grounded)
        self.assertEqual(world.me.boost, 33)
        self.assertAlmostEqual(world.me.forward.x, 1.0)
        self.assertEqual(world.latest_touch_index, 1)
        self.assertAlmostEqual(world.latest_touch_time, 2.5)
        self.assertFalse(world.boost_pads[0].is_active)
        self.assertEqual(len(world.ball.slices), 2)

    def test_bot_lifecycle_returns_controls(self):
        zero = Vector3()
        physics = Physics(
            location=Vector3(0, -1000, 17), rotation=Rotator(),
            velocity=Vector3(1000, 0, 0), angular_velocity=zero,
        )
        ball_physics = Physics(
            location=Vector3(0, 0, 93), rotation=Rotator(),
            velocity=zero, angular_velocity=zero,
        )
        packet = GamePacket(
            players=[PlayerInfo(physics=physics, air_state=AirState.OnGround, team=0, boost=50, player_id=0)],
            boost_pads=[BoostPadState(is_active=True)], balls=[BallInfo(physics=ball_physics)],
            match_info=MatchInfo(seconds_elapsed=1.0, match_phase=MatchPhase.Active, world_gravity_z=-650),
            teams=[],
        )
        prediction = BallPrediction(slices=[
            PredictionSlice(game_seconds=1.0 + i / 120, physics=ball_physics)
            for i in range(121)
        ])
        bot = AzureBot.__new__(AzureBot)
        bot.player_id = 0
        bot.team = 0
        bot.logger = SimpleNamespace(info=lambda *args: None, error=lambda *args: None)
        bot.field_info = SimpleNamespace(boost_pads=[FlatBoostPad(Vector3(1000, 0, 0), True)])
        bot.ball_prediction = prediction
        bot.initialize()
        controls = bot.get_output(packet)
        self.assertIsInstance(controls, ControllerState)
        self.assertIsNotNone(bot._action)


class StrategyTests(unittest.TestCase):
    def test_eta_increases_with_distance(self):
        car = _car(Vec3(), Vec3(1000, 0, 0))
        self.assertLess(estimate_eta(car, Vec3(1000, 0, 0)), estimate_eta(car, Vec3(5000, 0, 0)))

    def test_opponent_reach_prefers_closer_fast_car(self):
        slow = _car(Vec3(1000, 0, 0), Vec3(100, 0, 0))
        fast = _car(Vec3(500, 0, 0), Vec3(2000, 0, 0))
        target = Vec3(3000, 0, 0)
        self.assertLess(opponent_reach(fast, target), opponent_reach(slow, target))

    def test_aerial_budget_rejects_insufficient_boost(self):
        car = _car(Vec3(0, 0, 17), Vec3())
        car = replace(car, boost=5)
        self.assertFalse(aerial_possible(car, Vec3(0, 1000, 450), 1.0, False))

    def test_aerial_budget_accepts_nearby_high_ball(self):
        car = _car(Vec3(0, 400, 17), Vec3(0, 500, 0))
        car = replace(
            car, boost=100,
            forward=Vec3(0, 1, 0), right=Vec3(-1, 0, 0),
        )
        self.assertTrue(aerial_possible(car, Vec3(0, 900, 450), 1.6, True))


class ActionPriorityTests(unittest.TestCase):
    def test_kickoff_has_priority(self):
        world = _world(kickoff=True)
        self.assertEqual(choose_action(world).name, "kickoff")

    def test_airborne_recovery_has_priority(self):
        world = _world()
        world = _replace_world(world, me=_airborne_car())
        self.assertEqual(choose_action(world).name, "recovery")

    def test_ground_shot_is_selected_for_reachable_shot(self):
        world = _world()
        action = choose_action(world)
        self.assertEqual(action.name, "power_shot")

    def test_own_goal_prediction_selects_save(self):
        world = _world()
        slices = tuple(
            BallSlice(i / 120, Vec3(0, -4500 - 1200 * i / 120, 100), Vec3(0, -1200, 0))
            for i in range(91)
        )
        world = _replace_world(
            world,
            me=_car(Vec3(0, -4800, 17), Vec3()),
            ball=BallState(slices[0].location, slices[0].velocity, slices),
        )
        self.assertTrue(choose_action(world).name.startswith("save_"))


class KickoffTests(unittest.TestCase):
    def test_blue_center_kickoff_steers_toward_ball(self):
        world = _kickoff_world(team=0, location=Vec3(0, -4608, 17), forward=Vec3(0, 1, 0), right=Vec3(-1, 0, 0))
        controls = KickoffAction().tick(world)
        self.assertGreater(world.me.local(world.ball.location - world.me.location).x, 0)
        self.assertLessEqual(abs(controls.steer), 0.01)
        self.assertGreater(controls.throttle, 0)

    def test_orange_center_kickoff_steers_toward_ball(self):
        world = _kickoff_world(team=1, location=Vec3(0, 4608, 17), forward=Vec3(0, -1, 0), right=Vec3(1, 0, 0))
        controls = KickoffAction().tick(world)
        self.assertGreater(world.me.local(world.ball.location - world.me.location).x, 0)
        self.assertLessEqual(abs(controls.steer), 0.01)
        self.assertGreater(controls.throttle, 0)

    def test_kickoff_speed_flip_starts_toward_offset_target(self):
        world = _kickoff_world(
            team=0, location=Vec3(0, -4000, 17), forward=Vec3(0, 1, 0),
            right=Vec3(-1, 0, 0), velocity=Vec3(0, 800, 0),
        )
        action = KickoffAction()
        action.tick(world)
        self.assertTrue(action._speed_flipped)
        self.assertIsNotNone(action._speed_flip)
        self.assertGreater(action._speed_flip.direction.y, 0.9)

    def test_kickoff_action_finishes_on_new_ball_touch(self):
        world = _kickoff_world(
            team=0, location=Vec3(0, -3000, 17), forward=Vec3(0, 1, 0),
            right=Vec3(-1, 0, 0), velocity=Vec3(0, 500, 0),
        )
        action = KickoffAction()
        action.tick(world)
        touched = _replace_world(world, time=0.1, latest_touch_time=0.1, latest_touch_index=0)
        controls = action.tick(touched)
        self.assertTrue(action.finished)
        self.assertEqual(
            (controls.throttle, controls.steer, controls.pitch, controls.yaw, controls.roll, controls.jump),
            (0.0, 0.0, 0.0, 0.0, 0.0, False),
        )

    def test_recent_ball_touch_does_not_restart_kickoff(self):
        world = _world(kickoff=True)
        world = _replace_world(world, time=10.0, latest_touch_time=9.9, latest_touch_index=0)
        self.assertNotEqual(choose_action(world).name, "kickoff")

    def test_kickoff_runs_even_without_ball_prediction(self):
        world = _world(kickoff=True)
        world = _replace_world(world, ball=BallState(world.ball.location, world.ball.velocity, ()))
        self.assertEqual(choose_action(world).name, "kickoff")

    def test_kickoff_close_finish_creates_one_dodge(self):
        world = _kickoff_world(
            team=0, location=Vec3(0, -500, 17), forward=Vec3(0, 1, 0),
            right=Vec3(-1, 0, 0), velocity=Vec3(0, 700, 0),
        )
        action = KickoffAction()
        action._diagonal = False
        action._speed_flipped = True
        action._speed_flip = SpeedFlipAction(Vec3(0, 1, 0))
        action._speed_flip.finished = True
        action._time_on_ground = 0.2
        controls = action.tick(world)
        self.assertIsNotNone(action._final_dodge)
        self.assertFalse(action.finished)
        self.assertTrue(controls.jump)


class TimingTests(unittest.TestCase):
    def test_jump_shot_releases_for_exactly_three_frames(self):
        action = JumpShotAction(0.075, Vec3(0, 0, 93), Vec3(0, 5000, 250))
        action._jumped = True
        action._jump_elapsed = 0.05
        controls = [action.tick(_replace_world(_world(), time=i / 120, me=_airborne_car())) for i in range(10)]
        self.assertEqual([control.jump for control in controls[:3]], [False, False, False])
        self.assertTrue(controls[-1].jump)

    def test_double_jump_releases_then_reapplies_jump(self):
        action = DoubleJumpShotAction(1.0, Vec3(0, 0, 93), Vec3(0, 5000, 250))
        action._jumped = True
        action._jump_elapsed = 0.21
        controls = [action.tick(_replace_world(_world(), time=i / 120, me=_airborne_car())) for i in range(3)]
        self.assertEqual([control.jump for control in controls[:3]], [False, False, False])
        action._jump_elapsed = 0.24
        reapplied = action.tick(_replace_world(_world(), time=4 / 120, me=_airborne_car()))
        self.assertTrue(reapplied.jump)
        self.assertEqual((reapplied.pitch, reapplied.yaw, reapplied.roll), (0, 0, 0))


class BoostTests(unittest.TestCase):
    def test_full_boost_chooses_active_nearby_pad(self):
        car = _car(Vec3(), Vec3(2300, 0, 0))
        pads = (
            _pad(0, Vec3(0, 0, 0), True, False, 10),
            _pad(1, Vec3(1000, 0, 0), True, True, 0),
        )
        chosen = find_best_boost(car, pads)
        self.assertEqual(chosen.index, 1)


def _kickoff_world(
    *, team: int, location: Vec3, forward: Vec3, right: Vec3,
    velocity: Vec3 = Vec3(),
) -> World:
    car = CarState(
        0, team, location, velocity, Vec3(), forward, right, Vec3(0, 0, 1),
        100, True, False, False, False,
    )
    ball = BallState(Vec3(0, 0, 92.75), Vec3())
    return World(
        time=0.0, delta=1 / 120, gravity=-650, phase="MatchPhase.Kickoff",
        ball=ball, me=car, teammates=(), opponents=(), boost_pads=(),
        kickoff=True, demolition=False, latest_touch_time=-1.0,
        latest_touch_index=-1,
    )


def _world(*, kickoff: bool = False) -> World:
    car = _car(Vec3(0, -1000, 17), Vec3(1000, 0, 0))
    slices = tuple(
        BallSlice(i / 120, Vec3(0, 0, 93), Vec3())
        for i in range(361)
    )
    return World(
        time=0.0, delta=1 / 120, gravity=-650, phase="MatchPhase.Game",
        ball=BallState(Vec3(0, 0, 93), Vec3(), slices),
        me=car, teammates=(), opponents=(), boost_pads=(), kickoff=kickoff,
        demolition=False, latest_touch_time=-1.0, latest_touch_index=-1,
    )


def _replace_world(world: World, **changes) -> World:
    return replace(world, **changes)


def _airborne_car() -> CarState:
    return CarState(
        0, 0, Vec3(0, -1000, 300), Vec3(0, 1200, 0), Vec3(),
        Vec3(0, 1, 0), Vec3(1, 0, 0), Vec3(0, 0, -1),
        50, False, True, False, False,
    )


def _car(location: Vec3, velocity: Vec3) -> CarState:
    return CarState(0, 0, location, velocity, Vec3(), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), 50, True, False, False, False)


def _pad(index: int, location: Vec3, large: bool, active: bool, timer: float):
    from azure.field import BoostPad
    return BoostPad(index, location, large, active, timer)


if __name__ == "__main__":
    unittest.main()
