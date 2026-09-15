"""The scheduler must not be killable by a bad cycle.

A scheduler that exits on the first exception is worse than no scheduler,
because it looks healthy right up until the day it silently stopped.
"""
from __future__ import annotations

import json

from mlbedge import scheduler as S


def test_a_failing_cycle_does_not_kill_the_loop(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "HEARTBEAT", tmp_path / "hb.json")
    monkeypatch.setattr(S, "LOGFILE", tmp_path / "s.log")
    sch = S.Scheduler(seasons=(2025,))

    def boom():
        raise RuntimeError("simulated API outage")

    sch.cycle = boom
    assert sch.run_once() == -1, "a failed cycle reports, it does not raise"

    hb = json.loads((tmp_path / "hb.json").read_text())
    assert "simulated API outage" in hb["last_error"]
    assert "updated_at" in hb


def test_a_quiet_cycle_is_recorded_as_success(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "HEARTBEAT", tmp_path / "hb.json")
    monkeypatch.setattr(S, "LOGFILE", tmp_path / "s.log")
    sch = S.Scheduler(seasons=(2025,))
    sch.cycle = lambda: 0          # no games in window is not a failure
    assert sch.run_once() == 0

    hb = json.loads((tmp_path / "hb.json").read_text())
    assert hb.get("last_success")
    assert hb.get("last_error") is None


def test_serving_window_stays_inside_the_training_distribution():
    """Models are fitted on snapshots 30-45 minutes before first pitch, so the
    scheduler must not ask them to price a game hours out."""
    assert S.LEAD_MIN_MIN >= 15
    assert S.LEAD_MAX_MIN <= 120, "serving far outside the fitted lead times"
