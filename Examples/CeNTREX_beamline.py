import os

import matplotlib.pyplot as plt
import numpy as np
from CeNTREX_trajectories import (
    CircularAperture,
    Coordinates,
    ElectrostaticQuadrupoleLens,
    Gravity,
    PropagationType,
    RectangularAperture,
    Section,
    TlF,
    Velocities,
    propagate_trajectories,
    PropagationOptions,
)
from rich.progress import (
    Progress,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    BarColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    ProgressColumn,
)

import asciichartpy as acp
from rich.panel import Panel
from rich.live import Live
from rich.console import Group


def get_panel(data, title, height=10, format="{:8.5f}"):
    return Panel(acp.plot(data, {"height": height, "format": format}), title=title)


class TaskSpeed(ProgressColumn):
    def render(self, task):
        if task.speed is None:
            return ""
        elif task.speed >= 0.1:
            return f"{( task.speed or 0 ):.1f}/s"
        else:
            return f"{( 1 / task.speed or 0):.1f} s/i"


total_runs = 10
n_trajectories = 4_000_000
options = PropagationOptions(verbose=False)
gravity = Gravity(0, -9.81, 0)
particle = TlF()
# Ls = np.linspace(0.1, 0.7, 5)
Ls = [0, 0.6]
Vs = np.linspace(0, 30e3, 16)
# Vs = [30e3]
# Vs = [27.6e3]
# Vs = [24e3]

fourK = Section(
    "4K shield",
    [CircularAperture(0, 0, 1.75 * 0.0254, 0.0254 / 2)],
    0,
    (1.75 + 0.25) * 0.0254,
    False,
    propagation_type=PropagationType.ballistic,
)
fourtyK = Section(
    "40K shield",
    [CircularAperture(0, 0, fourK.stop + 1.25 * 0.0254, 0.0254 / 2)],
    fourK.stop,
    fourK.stop + (1.25 + 0.25) * 0.0254,
    False,
    PropagationType.ballistic,
)
bbexit = Section(
    "Beamsource Exit",
    [CircularAperture(0, 0, fourtyK.stop + 2.5 * 0.0254, 2 * 0.0254)],
    fourtyK.stop,
    fourtyK.stop + (2.5 + 0.75) * 0.0254,
    False,
    propagation_type=PropagationType.ballistic,
)
spa = Section(
    "State Prep A",
    [
        CircularAperture(0, 0, bbexit.stop + 19.6 * 0.0254, 1.75 * 2 / 0.0254),
        CircularAperture(
            0, 0, bbexit.stop + (19.6 + 0.375 + 9.625) * 0.0254, 1.75 / 2 * 0.0254,
        ),
    ],
    bbexit.stop,
    bbexit.stop + (19.6 + 0.375 + 9.625 + 0.375) * 0.0254,
    False,
    propagation_type=PropagationType.ballistic,
)

if __name__ == "__main__":
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description} : {task.fields[value]}", justify="right"),
        BarColumn(bar_width=None),
        TaskProgressColumn(show_speed=True),
        TaskSpeed(),
        TextColumn("{task.completed} of {task.total}"),
        "•",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    panel = Panel("")
    group = Group(panel, progress,)
    data = [[]]
    throughput_list, particles_detected_list = [], []
    with Live(group, refresh_per_second=1) as live:
        task_total = progress.add_task("[red]Repeating", total=total_runs, value=None)
        for repeat in range(total_runs):
            progress.update(task_total, advance=0, value=f"{repeat+1}")
            task_L = progress.add_task("[blue]Length scan", total=len(Ls), value=None)
            origin = Coordinates(
                np.random.randn(n_trajectories) * 1.5e-3,
                np.random.randn(n_trajectories) * 1.5e-3,
                np.zeros(n_trajectories),
            )
            velocities = Velocities(
                np.random.randn(n_trajectories) * 39.4,
                np.random.randn(n_trajectories) * 39.4,
                np.random.randn(n_trajectories) * 16 + 184,
            )
            throughput = {}
            particles_detected = {}
            for L in Ls:
                progress.update(task_L, advance=0, value=f"{L} m")
                task_V = progress.add_task(
                    "[green]Voltage scan", total=len(Vs), value=None
                )
                _throughput = []
                _particles_detected = []
                for V in Vs:
                    progress.update(task_V, advance=0, value=f"{V:.1f} V")
                    if L == 0:
                        V = 0
                    eql = ElectrostaticQuadrupoleLens(
                        "Electrostatic Lens",
                        [],
                        1.01,
                        1.01 + L,
                        V,
                        1.75 * 0.0254 / 2,
                        False,
                    )
                    interaction = Section(
                        "Interaction",
                        [RectangularAperture(0, 0, eql.stop + 0.82 + 0.1, 0.03, 0.03)],
                        eql.stop + 0.82,
                        eql.stop + 0.82 + 3,
                        True,
                        propagation_type=PropagationType.ballistic,
                        force=None,
                    )
                    det = Section(
                        "Detection",
                        [
                            RectangularAperture(
                                0, 0, interaction.stop + 1.01 + 0.001, 0.03, 0.03
                            )
                        ],
                        interaction.stop + 1.01,
                        interaction.stop + 1.01 + 0.01,
                        False,
                        PropagationType.ballistic,
                        force=None,
                    )
                    # sections = [fourK, fourtyK, bbexit, spa, interaction, det]
                    sections = [fourK, fourtyK, bbexit, spa, eql, interaction, det]
                    section_data, trajectories = propagate_trajectories(
                        sections,
                        origin,
                        velocities,
                        particle,
                        gravity=gravity,
                        options=options,
                    )

                    section = section_data[-1]
                    thru = (
                        section.nr_trajectories - section.nr_collisions
                    ) / n_trajectories
                    print(f"L = {L}, V = {V:.1f} => througput = {thru:.5%}")
                    _particles_detected.append(
                        section.nr_trajectories - section.nr_collisions
                    )
                    _throughput.append(
                        (section.nr_trajectories - section.nr_collisions)
                        / n_trajectories
                    )
                    progress.update(task_V, advance=1)
                    if L == 0:
                        break
                    width = int(0.9 * os.get_terminal_size().columns // len(Vs))
                    for _ in range(width):
                        data[-1].append(
                            _particles_detected[-1] / particles_detected[0][0]
                        )
                    if len(_throughput) == len(Vs):
                        data.append([])
                    group.renderables[0] = get_panel(
                        data, title="Gain ", height=15, format="{:5.1f}"
                    )
                particles_detected[L] = np.asarray(_particles_detected)
                throughput[L] = np.asarray(_throughput)
                progress.update(task_L, advance=1)
                progress.remove_task(task_V)
            throughput_list.append(throughput)
            particles_detected_list.append(particles_detected)
            progress.update(task_total, advance=1)
            progress.remove_task(task_L)

    fig, ax = plt.subplots(figsize=(8, 5))
    for throughput, particles_detected in zip(throughput_list, particles_detected_list):
        for L in throughput.keys():
            if L == 0:
                continue

            gain = particles_detected[L] / particles_detected[0][0]
            sigma = (gain) ** 2 * (
                particles_detected[L] / particles_detected[L] ** 2
                + particles_detected[0][0] / particles_detected[0][0] ** 2
            )
            sigma = np.sqrt(sigma)

            p = ax.errorbar(Vs, gain, yerr=sigma, lw=2, label=f"L={L:.1f}",)
            ax.fill_between(
                Vs, gain - sigma, gain + sigma, color=p[0].get_color(), alpha=0.5,
            )
    ax.grid(True)
    ax.legend()
    ax.set_yscale("log")

    ys = []
    yerrs = []
    for throughput, particles_detected in zip(throughput_list, particles_detected_list):
        for L in throughput.keys():
            if L == 0:
                continue

            gain = particles_detected[L] / particles_detected[0][0]
            sigma = (gain) ** 2 * (
                particles_detected[L] / particles_detected[L] ** 2
                + particles_detected[0][0] / particles_detected[0][0] ** 2
            )
            sigma = np.sqrt(sigma)
            ys.append(gain)
            yerrs.append(sigma)

    sigmas = np.std(ys, axis=0) / np.sqrt(len(throughput_list))
    ymeans = np.mean(ys, axis=0)
    # print(ys.shape, Vs.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    p = ax.errorbar(Vs, ymeans, yerr=sigmas, lw=2, label=f"L={L:.1f}",)
    ax.fill_between(
        Vs, ymeans - sigmas, ymeans + sigmas, color=p[0].get_color(), alpha=0.5,
    )
    ax.grid(True)
    ax.legend()
    ax.set_yscale("log")
    plt.show()
