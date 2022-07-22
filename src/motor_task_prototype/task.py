from psychopy.gui import DlgFromDict
from psychopy.visual.window import Window
from psychopy.visual.circle import Circle
from psychopy.visual.elementarray import ElementArrayStim
from psychopy.visual.shape import BaseShapeStim, ShapeStim
from psychopy.visual.basevisual import BaseVisualStim
from psychopy.visual.textbox2 import TextBox2
from psychopy.colors import Color, colors

colors.pop("none")
from psychopy.sound import Sound
from psychopy.clock import Clock
from psychopy.data import TrialHandlerExt, importConditions
from psychopy import core
from psychopy.event import Mouse, xydist
from psychopy.hardware.keyboard import Keyboard
from dataclasses import dataclass
from typing import List, Tuple, Union
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from enum import Enum
import numpy as np
from motor_task_prototype import geometry as mtpgeom
from motor_task_prototype import analysis as mtpanalysis


def make_cursor(window: Window) -> ShapeStim:
    return ShapeStim(
        window,
        lineColor="black",
        pos=(0, 0),
        lineWidth=5,
        vertices=[
            (-0.01, 0.00),
            (0.01, 0.00),
            (0.00, 0.00),
            (0.00, -0.01),
            (0.00, 0.01),
        ],
        closeShape=False,
    )


def make_targets(
    window: Window,
    n_circles: int,
    radius: float,
    point_radius: float,
    center_point_radius: float,
) -> ElementArrayStim:
    return ElementArrayStim(
        window,
        units="height",
        fieldShape="circle",
        nElements=n_circles + 1,
        sizes=[2.0 * point_radius] * n_circles + [2.0 * center_point_radius],
        xys=mtpgeom.points_on_circle(n_circles, radius, include_centre=True),
        elementTex=None,
        elementMask="circle",
    )


def select_target(targets: ElementArrayStim, index: int = None) -> None:
    c = np.array([[0.1, 0.1, 0.1]] * targets.nElements)
    if index is not None:
        c[index][0] = 1
        c[index][1] = -1
        c[index][2] = -1
    targets.setColors(c, colorSpace="rgb")


def draw_and_flip(win: Window, drawables: List[BaseVisualStim], kb: Keyboard) -> None:
    for drawable in drawables:
        drawable.draw()
    if kb.getKeys(["escape"]):
        core.quit()
    win.flip()


MotorTaskSettingsDict = TypedDict(
    "MotorTaskSettingsDict",
    {
        "num_targets": int,
        "target_order": str,
        "target_indices": np.ndarray,
        "target_duration": float,
        "inter_target_duration": float,
        "target_distance": float,
        "target_size": float,
        "central_target_size": float,
        "play_sound": bool,
        "show_cursor": bool,
        "show_cursor_path": bool,
        "cursor_rotation": float,
    },
)


@dataclass
class MotorTaskTargetResult:
    target_index: int
    target_position: Tuple[float, float]
    mouse_times: List[float]
    mouse_positions: List[Tuple[float, float]]


def get_settings_from_user(
    settings: MotorTaskSettingsDict = None,
) -> MotorTaskSettingsDict:
    if settings is None:
        settings = {
            "num_targets": 8,
            "target_order": "clockwise",
            "target_indices": [],
            "target_duration": 5,
            "inter_target_duration": 1,
            "target_distance": 0.4,
            "target_size": 0.04,
            "central_target_size": 0.02,
            "play_sound": True,
            "show_cursor": True,
            "show_cursor_path": True,
            "cursor_rotation": 0.0,
        }
    order_of_targets = [settings["target_order"]]
    for target_order in ["clockwise", "anti-clockwise", "random"]:
        if target_order != order_of_targets[0]:
            order_of_targets.append(target_order)
    dialog = DlgFromDict(settings, title="Motor task settings", sortKeys=False)
    if not dialog.OK:
        core.quit()
    settings["cursor_rotation"] = settings["cursor_rotation"] * (2.0 * np.pi / 360.0)
    return settings


class MotorTask:
    trials: TrialHandlerExt

    def __init__(self, settings: MotorTaskSettingsDict):
        settings["target_indices"] = np.array(range(settings["num_targets"]))
        if settings["target_order"] == "anti-clockwise":
            settings["target_indices"] = np.flip(settings["target_indices"])
        elif settings["target_order"] == "anti-clockwise":
            rng = np.random.default_rng()
            rng.shuffle(settings["target_indices"])
        self.trials = TrialHandlerExt([settings], 1, originPath=-1)

    def run(self, win: Window) -> TrialHandlerExt:
        trial = self.trials.getCurrentTrial()
        print(trial)
        mouse = Mouse(visible=False)
        clock = Clock()
        kb = Keyboard()
        targets: ElementArrayStim = make_targets(
            win,
            trial["num_targets"],
            trial["target_distance"],
            trial["target_size"],
            trial["central_target_size"],
        )
        drawables: List[Union[BaseVisualStim, ElementArrayStim]] = [targets]
        cursor = make_cursor(win)
        if trial["show_cursor"]:
            drawables.append(cursor)
        rotated_point = mtpgeom.RotatedPoint(trial["cursor_rotation"])
        cursor_path = ShapeStim(
            win, vertices=[(0, 0)], lineColor="white", closeShape=False
        )
        if trial["show_cursor_path"]:
            drawables.append(cursor_path)
        for target_index in trial["target_indices"]:
            self.trials.addData("target_index", targets.xys[target_index])
            self.trials.addData("target_pos", targets.xys[target_index])
            select_target(targets, None)
            cursor_path.vertices = [(0, 0)]
            cursor.setPos((0.0, 0.0))
            clock.reset()
            while clock.getTime() < trial["inter_target_duration"]:
                draw_and_flip(win, drawables, kb)
            select_target(targets, target_index)
            if trial["play_sound"]:
                Sound("A", secs=0.3, blockSize=512).play()
            mouse_pos = (0.0, 0.0)
            dist = xydist(mouse_pos, targets.xys[target_index])
            mouse_times = [0]
            mouse_positions = [mouse_pos]
            mouse.setPos(mouse_pos)
            draw_and_flip(win, drawables, kb)
            clock.reset()
            mouse.setPos(mouse_pos)
            win.recordFrameIntervals = True
            while (
                dist > trial["target_size"]
                and clock.getTime() < trial["target_duration"]
            ):
                mouse_pos = rotated_point(mouse.getPos())
                if trial["show_cursor"]:
                    cursor.setPos(mouse_pos)
                mouse_times.append(clock.getTime())
                mouse_positions.append(mouse_pos)
                if trial["show_cursor_path"]:
                    cursor_path.vertices = mouse_positions
                dist = xydist(mouse_pos, targets.xys[target_index])
                draw_and_flip(win, drawables, kb)
            win.recordFrameIntervals = False
            self.trials.addData("timestamps", mouse_positions)
            self.trials.addData("mouse_positions", mouse_positions)
        if win.nDroppedFrames > 0:
            print(f"Warning: dropped {win.nDroppedFrames} frames")
        win.flip()
        return self.trials

    def display_results(
        self, win: Window, settings: MotorTaskSettingsDict, results: TrialHandlerExt
    ) -> None:
        clock = Clock()
        kb = Keyboard()
        drawables: List[BaseVisualStim] = []
        for result, color in zip(results, colors):
            drawables.append(
                Circle(
                    win,
                    radius=settings["target_size"],
                    pos=result.target_position,
                    fillColor=color,
                )
            )
            drawables.append(
                ShapeStim(
                    win,
                    vertices=result.mouse_positions,
                    lineColor=color,
                    closeShape=False,
                    lineWidth=3,
                )
            )
            reac, move = mtpanalysis.reaction_movement_times(
                result.mouse_times, result.mouse_positions
            )
            dist = mtpanalysis.distance(result.mouse_positions)
            rmse = mtpanalysis.rmse(result.mouse_positions, result.target_position)
            if result.target_position[0] > 0:
                text_pos = result.target_position[0] + 0.16, result.target_position[1]
            else:
                text_pos = result.target_position[0] - 0.16, result.target_position[1]
            drawables.append(
                TextBox2(
                    win,
                    f"Reaction: {reac:.3f}s\nMovement: {move:.3f}s\nDistance: {dist:.3f}\nRMSE: {rmse:.3f}",
                    pos=text_pos,
                    color=color,
                    alignment="center",
                    letterHeight=0.02,
                )
            )
        clock.reset()
        while clock.getTime() < 30:
            draw_and_flip(win, drawables, kb)
