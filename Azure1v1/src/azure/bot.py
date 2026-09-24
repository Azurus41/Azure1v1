"""RLBot adapter and action lifecycle."""

from __future__ import annotations

from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot

from .field import BoostPad
from .model import BoostTracker, snapshot
from .strategy import choose_action
from .vector import vec


class AzureBot(Bot):
    """Party Cannon-inspired 1v1 action bot."""

    def initialize(self) -> None:
        pads = tuple(
            BoostPad(index, vec(pad.location), bool(pad.is_full_boost), False, 0.0)
            for index, pad in enumerate(self.field_info.boost_pads)
        )
        self._boost_tracker = BoostTracker(pads)
        self._previous_time = 0.0
        self._last_touch_time = -1.0
        self._was_kickoff = False
        self._action = None
        self.logger.info(f"Azure initialized: player={self.player_id}, team={self.team}")

    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.balls:
            self._action = None
            return ControllerState()
        prediction = self.ball_prediction
        world, self._boost_tracker = snapshot(
            packet,
            player_id=self.player_id,
            team=self.team,
            previous_time=self._previous_time,
            tracker=self._boost_tracker,
            prediction=prediction,
        )
        self._previous_time = world.time

        if world.kickoff and not self._was_kickoff:
            self._action = None
        self._was_kickoff = world.kickoff

        action = self._action
        if action is not None:
            touched = world.latest_touch_time > self._last_touch_time + 1e-6
            if action.finished or world.demolition or (action.interruptible and touched):
                action = None
                self._action = None
        if action is None:
            action = choose_action(world)
            self._action = action
        self._last_touch_time = max(self._last_touch_time, world.latest_touch_time)
        if action is None:
            return ControllerState()
        controls = action.tick(world)
        if not isinstance(controls, ControllerState):
            self.logger.error("Action returned invalid controls: %s", action.name)
            return ControllerState()
        self._action = None if action.finished else action
        return controls

