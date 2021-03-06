import motor_task_prototype.geometry as geometry
import numpy as np


def test_equidistant_angles() -> None:
    assert np.allclose(geometry.equidistant_angles(1) / np.pi, [0])
    assert np.allclose(geometry.equidistant_angles(2) / np.pi, [0, 1])
    assert np.allclose(
        geometry.equidistant_angles(3) / np.pi, [0, 2.0 / 3.0, 4.0 / 3.0]
    )
    assert np.allclose(geometry.equidistant_angles(4) / np.pi, [0, 0.5, 1.0, 1.5])


def test_points_on_circle() -> None:
    assert np.allclose(geometry.points_on_circle(1, 0.2), [(0.0, 0.2)])
    assert np.allclose(
        geometry.points_on_circle(1, 0.2, include_centre=True), [(0.0, 0.2), (0.0, 0.0)]
    )
    assert np.allclose(geometry.points_on_circle(2, 0.5), [(0.0, 0.5), (0.0, -0.5)])
    assert np.allclose(
        geometry.points_on_circle(4, 1.0), [(0, 1), (1, 0), (0, -1), (-1, 0)]
    )


def test_rotate_point() -> None:
    pivots = [(0, 0), (0.3, 0.2), (-0.3, 0.4), (0.6, -0.4), (-1, -1)]
    points = [(1, 0.2), (-1, 0.2), (-0.7, -0.5), (0.7, 0.0), (0.0, 0.6), (0, 0)]
    # rotate by zero or 360 degrees around any pivot point returns original point
    for angle in [0, 2 * np.pi]:
        for pivot in pivots:
            for point in points:
                assert np.allclose(geometry.rotate_point(point, angle, pivot), point)
    # rotating a point around itself returns the original point
    for angle in [0, -0.1, 0.3, 0.66, np.pi, -np.pi, 4.5]:
        for point in points:
            assert np.allclose(geometry.rotate_point(point, angle, point), point)
    # rotating by +180 degrees is equivalent to rotating by -180 degrees
    for pivot in pivots:
        for point in points:
            assert np.allclose(
                geometry.rotate_point(point, np.pi, pivot),
                geometry.rotate_point(point, -np.pi, pivot),
            )
    # todo: non-trivial tests of rotations


def test_rotated_point() -> None:
    for angle in [0, -0.1, 0.3, 0.66, np.pi, -np.pi, 4.5]:
        rp = geometry.RotatedPoint(angle)
        for point in [
            (1, 0.2),
            (-1, 0.2),
            (-0.7, -0.5),
            (0.7, 0.0),
            (0.0, 0.6),
            (0, 0),
        ]:
            p1 = rp(point)
            p2 = geometry.rotate_point(point, angle, (0, 0))
            assert np.allclose([p1], [p2])
