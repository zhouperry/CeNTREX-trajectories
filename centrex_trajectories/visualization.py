from typing import List, Optional, Union

import matplotlib.axes as axes
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .beamline_objects import CircularAperture, ODESection, RectangularAperture, Section

__all__ = ["plot_beamline"]


def plot_beamline(
    sections: List[Union[Section, ODESection]],
    ax: Optional[axes.Axes] = None,
    facecolor: str = "C0",
    edgecolor: str = "k",
    alpha: float = 0.5,
    axis: str = "x",
):
    objects = []
    for section in sections:
        objects.extend(section.objects)

    height = 0.0
    for obj in objects:
        if isinstance(obj, CircularAperture):
            if axis == "x":
                if height <= (obj.x + obj.r) * 2:
                    height = (obj.x + obj.r) * 2
            elif axis == "y":
                if height <= (obj.y + obj.r) * 2:
                    height = (obj.y + obj.r) * 2
        if isinstance(obj, RectangularAperture):
            if axis == "x":
                if height <= (obj.x + obj.wx / 2) * 2:
                    height = (obj.x + obj.wx / 2) * 2
            elif axis == "y":
                if height <= (obj.y + obj.wy / 2) * 2:
                    height = (obj.y + obj.wy / 2) * 2
    patches = [
        Rectangle((section.start, -height), section.stop - section.start, 2 * height)
        for section in sections
    ]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(patches, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for obj in objects:
        if isinstance(obj, CircularAperture):
            if axis == "x":
                ax.plot(
                    [obj.z, obj.z], [obj.x - obj.r, obj.x + obj.r], lw=4, color="C3"
                )
            elif axis == "y":
                ax.plot(
                    [obj.z, obj.z], [obj.y - obj.r, obj.y + obj.r], lw=4, color="C3"
                )
        if isinstance(obj, RectangularAperture):
            if axis == "x":
                ax.plot(
                    [obj.z, obj.z],
                    [obj.x - obj.wx / 2, obj.x + obj.wx / 2],
                    lw=4,
                    color="C3",
                )
            elif axis == "y":
                ax.plot(
                    [obj.z, obj.z],
                    [obj.y - obj.wy / 2, obj.y + obj.wy / 2],
                    lw=4,
                    color="C3",
                )

    # Add collection to axes
    ax.add_collection(pc)

    dx = sections[-1].stop - sections[0].start
    x0 = sections[0].start
    ax.set_xlim(x0 - 0.1 * dx, x0 + 1.1 * dx)
    ax.set_ylim(-height * 1.1, height * 1.1)

    ax.set_xlabel("z [m]")
    ax.set_ylabel(f"{axis} [m]")

    return ax
