import numpy as np

from centrex_trajectories.data_structures import Coordinates, Force, Velocities
from centrex_trajectories.propagation_ballistic import (
    calculate_time_ballistic,
    propagate_ballistic,
)


def test_propagate_ballistic():
    t = np.arange(1e-3, 1.1e-2, 1e-3)
    origin = Coordinates(
        x=np.linspace(-1e-3, 1e-3, len(t)),
        y=np.linspace(-1e-3, 1e-3, len(t)),
        z=np.zeros(len(t)),
    )
    velocities = Velocities(
        vx=np.linspace(-1, 1, len(t)),
        vy=np.linspace(-1, 1, len(t)),
        vz=np.linspace(180, 190, len(t)),
    )
    gravity = Force(0, -9.81, 0)
    stop_coords, stop_vels = propagate_ballistic(t, origin, velocities, gravity)
    true_coords = Coordinates(
        x=np.array(
            [
                -0.002,
                -0.00233333,
                -0.00222222,
                -0.00166667,
                -0.00066667,
                0.00077778,
                0.00266667,
                0.005,
                0.00777778,
                0.011,
            ]
        ),
        y=np.array(
            [
                -0.0020049,
                -0.00235295,
                -0.00226637,
                -0.00174515,
                -0.00078929,
                0.0006012,
                0.00242632,
                0.00468608,
                0.00738047,
                0.0105095,
            ]
        ),
        z=np.array(
            [
                0.18,
                0.36222222,
                0.54666667,
                0.73333333,
                0.92222222,
                1.11333333,
                1.30666667,
                1.50222222,
                1.7,
                1.9,
            ]
        ),
    )
    true_vels = Velocities(
        vx=np.array(
            [
                -1.0,
                -0.77777778,
                -0.55555556,
                -0.33333333,
                -0.11111111,
                0.11111111,
                0.33333333,
                0.55555556,
                0.77777778,
                1.0,
            ]
        ),
        vy=np.array(
            [
                -1.00981,
                -0.79739778,
                -0.58498556,
                -0.37257333,
                -0.16016111,
                0.05225111,
                0.26466333,
                0.47707556,
                0.68948778,
                0.9019,
            ]
        ),
        vz=np.array(
            [
                180.0,
                181.11111111,
                182.22222222,
                183.33333333,
                184.44444444,
                185.55555556,
                186.66666667,
                187.77777778,
                188.88888889,
                190.0,
            ]
        ),
    )
    assert np.allclose(stop_coords.x, true_coords.x)
    assert np.allclose(stop_coords.y, true_coords.y)
    assert np.allclose(stop_coords.z, true_coords.z)
    assert np.allclose(stop_vels.vx, true_vels.vx)
    assert np.allclose(stop_vels.vy, true_vels.vy)
    assert np.allclose(stop_vels.vz, true_vels.vz)


def test_calculate_time_ballistic():
    x = np.arange(-10, 11, 1)
    v = np.arange(180, 201, 1)
    a = 0.0

    t_ballistic = calculate_time_ballistic(x, v, a)
    assert np.isnan(t_ballistic[:10]).sum() == 10

    true_t = np.array(
        [
            0.0,
            0.0052356,
            0.01041667,
            0.01554404,
            0.02061856,
            0.02564103,
            0.03061224,
            0.03553299,
            0.04040404,
            0.04522613,
            0.05,
        ]
    )
    assert np.allclose(t_ballistic[10:], true_t)

    x = 1
    v = 100
    a = 100

    t_ballistic = calculate_time_ballistic(x, v, a)

    assert calculate_time_ballistic(1, 1, -1 / 2) == 2.0
    assert calculate_time_ballistic(1, 1, 1 / 2) == 0.8284271247461903
