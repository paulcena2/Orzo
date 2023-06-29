
from orzo.window import intersection


def test_intersection():

    # Diagonal up into box
    assert intersection(
        [1, 1, 1],
        [-1, -1, -1],
        [0, 0, 0],
        [2, 2, 2]
    )

    # Misses box
    assert not intersection(
        [0, 0, 1],
        [-5, 0, 0],
        [0, 0, 0],
        [2, 2, 2]
    )
