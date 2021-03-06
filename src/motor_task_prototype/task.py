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


class TargetOrder(Enum):
    CLOCKWISE = "Clockwise"
    ANTICLOCKWISE = "Anti-clockwise"
    RANDOM = "Random"


@dataclass
class MotorTaskSettings:
    n_points: int = 8
    order_of_targets: TargetOrder = TargetOrder.CLOCKWISE
    time_per_point: float = 5
    time_between_points: float = 2
    outer_radius: float = 0.45
    point_radius: float = 0.04
    center_point_radius: float = 0.02
    play_sound: bool = True
    show_cursor: bool = True
    show_cursor_line: bool = True
    rotate_cursor_radians: float = 0


MotorTaskSettingsDict = TypedDict(
    "MotorTaskSettingsDict",
    {
        "Number of targets": int,
        "Order of targets": List[str],
        "Target Duration (secs)": float,
        "Interval between targets (secs)": float,
        "Distance of targets (screen height)": float,
        "Size of targets (screen height)": float,
        "Size of center point (screen height)": float,
        "Audible tone on target display": bool,
        "Display cursor": bool,
        "Display cursor path": bool,
        "Rotate cursor (degrees)": float,
    },
)


@dataclass
class MotorTaskTargetResult:
    target_index: int
    target_position: Tuple[float, float]
    mouse_times: List[float]
    mouse_positions: List[Tuple[float, float]]


def get_settings_from_user(
    settings: MotorTaskSettings = MotorTaskSettings(),
) -> MotorTaskSettings:
    order_of_targets = [settings.order_of_targets.value]
    for target_order in TargetOrder:
        if target_order != settings.order_of_targets:
            order_of_targets.append(target_order.value)
    dialog_dict: MotorTaskSettingsDict = {
        "Number of targets": settings.n_points,
        "Order of targets": order_of_targets,
        "Target Duration (secs)": settings.time_per_point,
        "Interval between targets (secs)": settings.time_between_points,
        "Distance of targets (screen height)": settings.outer_radius,
        "Size of targets (screen height)": settings.point_radius,
        "Size of center point (screen height)": settings.center_point_radius,
        "Audible tone on target display": settings.play_sound,
        "Display cursor": settings.show_cursor,
        "Display cursor path": settings.show_cursor_line,
        "Rotate cursor (degrees)": settings.rotate_cursor_radians,
    }
    dialog = DlgFromDict(dialog_dict, title="Motor task settings", sortKeys=False)
    if dialog.OK:
        settings.n_points = dialog_dict["Number of targets"]
        settings.order_of_targets = TargetOrder(dialog_dict["Order of targets"])
        settings.time_per_point = dialog_dict["Target Duration (secs)"]
        settings.time_between_points = dialog_dict["Interval between targets (secs)"]
        settings.outer_radius = dialog_dict["Distance of targets (screen height)"]
        settings.point_radius = dialog_dict["Size of targets (screen height)"]
        settings.center_point_radius = dialog_dict[
            "Size of center point (screen height)"
        ]
        settings.play_sound = dialog_dict["Audible tone on target display"]
        settings.show_cursor = dialog_dict["Display cursor"]
        settings.show_cursor_line = dialog_dict["Display cursor path"]
        settings.rotate_cursor_radians = dialog_dict["Rotate cursor (degrees)"] * (
            2.0 * np.pi / 360.0
        )
    return settings


class MotorTask:
    settings: MotorTaskSettings
    target_indices: np.ndarray

    def __init__(self, settings: MotorTaskSettings = MotorTaskSettings()):
        self.settings = settings
        self.target_indices = np.array(range(self.settings.n_points))
        if self.settings.order_of_targets == TargetOrder.ANTICLOCKWISE:
            self.target_indices = np.flip(self.target_indices)
        elif self.settings.order_of_targets == TargetOrder.RANDOM:
            rng = np.random.default_rng()
            rng.shuffle(self.target_indices)

    def run(self, win: Window) -> List[MotorTaskTargetResult]:
        results = []
        mouse = Mouse(visible=False)
        clock = Clock()
        kb = Keyboard()
        targets: ElementArrayStim = make_targets(
            win,
            self.settings.n_points,
            self.settings.outer_radius,
            self.settings.point_radius,
            self.settings.center_point_radius,
        )
        drawables: List[Union[BaseVisualStim, ElementArrayStim]] = [targets]
        cursor = make_cursor(win)
        if self.settings.show_cursor:
            drawables.append(cursor)
        rotated_point = mtpgeom.RotatedPoint(self.settings.rotate_cursor_radians)
        cursor_path = ShapeStim(
            win, vertices=[(0, 0)], lineColor="white", closeShape=False
        )
        if self.settings.show_cursor_line:
            drawables.append(cursor_path)
        for target_index in self.target_indices:
            result = MotorTaskTargetResult(
                target_index, targets.xys[target_index], [], []
            )
            select_target(targets, None)
            cursor_path.vertices = [(0, 0)]
            cursor.setPos((0.0, 0.0))
            clock.reset()
            while clock.getTime() < self.settings.time_between_points:
                draw_and_flip(win, drawables, kb)
            select_target(targets, target_index)
            if self.settings.play_sound:
                Sound("A", secs=0.3, blockSize=512).play()
            mouse_pos = (0.0, 0.0)
            dist = xydist(mouse_pos, targets.xys[target_index])
            result.mouse_times.append(0)
            result.mouse_positions.append(mouse_pos)
            mouse.setPos(mouse_pos)
            draw_and_flip(win, drawables, kb)
            clock.reset()
            mouse.setPos(mouse_pos)
            win.recordFrameIntervals = True
            while (
                dist > self.settings.point_radius
                and clock.getTime() < self.settings.time_per_point
            ):
                mouse_pos = rotated_point(mouse.getPos())
                if self.settings.show_cursor:
                    cursor.setPos(mouse_pos)
                result.mouse_times.append(clock.getTime())
                result.mouse_positions.append(mouse_pos)
                if self.settings.show_cursor_line:
                    cursor_path.vertices = result.mouse_positions
                dist = xydist(mouse_pos, targets.xys[target_index])
                draw_and_flip(win, drawables, kb)
            win.recordFrameIntervals = False
            if win.nDroppedFrames > 0:
                print(f"Warning: dropped {win.nDroppedFrames} frames")
            results.append(result)
        win.flip()
        return results

    def display_results(
        self, win: Window, results: List[MotorTaskTargetResult]
    ) -> None:
        clock = Clock()
        kb = Keyboard()
        drawables: List[BaseVisualStim] = []
        for result, color in zip(results, colors):
            drawables.append(
                Circle(
                    win,
                    radius=self.settings.point_radius,
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
