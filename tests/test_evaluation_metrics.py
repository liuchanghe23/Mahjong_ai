from mahjong_ai.evaluation.metrics import DecisionTiming, aggregate, extract_player_stats


def test_extracts_win_deal_in_riichi_and_draw_metrics() -> None:
    events = [
        {"type": "start_game"},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "reach", "actor": 0},
        {"type": "hora", "actor": 0, "target": 1, "deltas": [8000, -8000, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "start_kyoku", "scores": [33000, 17000, 25000, 25000]},
        {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [1000, -1000, -1000, 1000]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    timings = {pid: DecisionTiming(count=2, total_ns=2_000_000, max_ns=1_500_000) for pid in range(4)}

    players = extract_player_stats(events, (34000, 16000, 24000, 26000), (1, 4, 3, 2), timings)

    assert players[0].wins == 1
    assert players[0].ron_wins == 1
    assert players[0].riichi == 1
    assert players[0].draw_tenpai_known == 1
    assert players[1].deal_ins == 1
    assert players[1].deal_in_points == 8000
    assert all(player.rounds == 2 for player in players)


def test_zero_delta_exhaustive_draw_is_reported_unknown() -> None:
    events = [
        {"type": "start_kyoku", "scores": [25000] * 4},
        {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [0, 0, 0, 0]},
    ]
    timings = {pid: DecisionTiming() for pid in range(4)}

    players = extract_player_stats(events, (25000,) * 4, (1, 2, 3, 4), timings)

    assert all(player.draw_tenpai_unknown == 1 for player in players)


def test_aggregate_uses_player_round_denominators() -> None:
    events = [
        {"type": "start_kyoku", "scores": [25000] * 4},
        {"type": "hora", "actor": 0, "target": 0, "deltas": [6000, -2000, -2000, -2000]},
    ]
    timings = {pid: DecisionTiming() for pid in range(4)}
    players = extract_player_stats(events, (31000, 23000, 23000, 23000), (1, 2, 3, 4), timings)

    stats = aggregate("test", [players])

    assert stats.rounds == 4
    assert stats.win_rate_per_round == 0.25
    assert stats.tsumo_wins == 1
    assert stats.average_rank == 2.5

