# Azure 1v1

A pure-Python Rocket League bot for RLBot. It is a deterministic action-based
bot inspired primarily by the recovered **Party Cannon** C# source, with useful
prediction ideas from Noob Black.

## Capabilities

- Party-style time-aware shot selection and target geometry
- ground/power shots, jump shots, double-jump shots, and boosted aerials
- emergency saves, goal-line defense, and safe clears
- controllable dribbles and ball carrying
- full-boost selection with availability and teammate checks
- kickoff speed flip, dodges, half flips, wavedashes, and recovery
- turn-aware ground control, delayed boost, and deterministic 120 Hz actions

## Run

```powershell
python -m pip install -r requirements.txt
python run.py
```

Edit `rlbot.toml` to choose the car/team. The default configuration runs a
human blue car and Azure orange. For development, use `dev.toml` and
`python src/bot.py`; it leaves agent startup to you.

## Validate

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src
```

The `src/azure` package is the implementation. `src/bot.py` is only the RLBot
entry point. The old deleted `src/bot_backup.py` has deliberately not been
restored or modified.

## Tuning

Physics constants are in `src/azure/constants.py`. Strategy priorities and
thresholds are in `src/azure/strategy.py`; action timing is in
`src/azure/actions.py`.

Party Cannon is the main behavioral reference. This Python implementation is
not a literal source translation: it keeps the action model and control ideas
while using RLBot's native ball prediction, modern packet model, and a testable
Python architecture.
