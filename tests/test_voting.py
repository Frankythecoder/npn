import numpy as np
import pytest

from ml.ensemble.voting import (
    EnsembleResult,
    combine_matrix,
    combine_one,
    votes_required,
)


@pytest.mark.parametrize(
    "n,threshold,expected",
    [
        (4, 0.5, 2),
        (4, 0.75, 3),
        (4, 1.0, 4),
        (4, 0.25, 1),
        (4, 0.0, 1),
        (7, 0.5, 4),
        (3, 0.5, 2),
        (1, 0.5, 1),
    ],
)
def test_votes_required_uses_ceiling_with_a_floor_of_one(n, threshold, expected):
    assert votes_required(n, threshold) == expected


def test_combine_one_flags_at_the_threshold():
    result = combine_one(
        {"a": 1, "b": 1, "c": 0, "d": 0}, threshold=0.5
    )
    assert isinstance(result, EnsembleResult)
    assert result.is_anomaly is True
    assert result.votes_for == 2
    assert result.votes_total == 4
    assert result.votes_required == 2


def test_combine_one_does_not_flag_below_the_threshold():
    result = combine_one({"a": 1, "b": 0, "c": 0, "d": 0}, threshold=0.5)
    assert result.is_anomaly is False
    assert result.votes_for == 1


def test_combine_one_rejects_an_empty_roster():
    with pytest.raises(ValueError, match="at least one detector"):
        combine_one({}, threshold=0.5)


def test_as_dict_matches_the_result_contract():
    keys = set(combine_one({"a": 1, "b": 1}, threshold=0.5).as_dict())
    assert keys == {
        "is_anomaly",
        "votes_for",
        "votes_total",
        "votes_required",
        "threshold",
    }


def test_combine_matrix_matches_combine_one_row_by_row():
    flags = {
        "a": np.array([1, 0, 1, 0]),
        "b": np.array([1, 1, 0, 0]),
        "c": np.array([0, 1, 0, 0]),
        "d": np.array([0, 0, 0, 1]),
    }
    votes, is_anomaly = combine_matrix(flags, threshold=0.5)
    assert votes.tolist() == [2, 2, 1, 1]
    assert is_anomaly.tolist() == [1, 1, 0, 0]
    for i in range(4):
        single = combine_one({k: int(v[i]) for k, v in flags.items()}, 0.5)
        assert single.votes_for == votes[i]
        assert int(single.is_anomaly) == is_anomaly[i]
