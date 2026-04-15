from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .constants import SimulationConstants
from .curling import Curling, StoneThrow
from .enums import StoneColor


class Agent(Protocol):
    name: str

    def select_shot(self, game: Curling, color: StoneColor) -> StoneThrow:
        ...


def parse_color(value: str) -> StoneColor | None:
    normalized = value.lower()
    if normalized == "red":
        return StoneColor.RED
    if normalized == "yellow":
        return StoneColor.YELLOW
    if normalized == "random":
        return None
    raise argparse.ArgumentTypeError(f"Unsupported color '{value}'. Use red, yellow, or random.")


def build_constants(mode: str) -> SimulationConstants:
    if mode == "accurate":
        return SimulationConstants(time_intervals=0.02)
    return SimulationConstants(time_intervals=(0.3, 0.1, 0.05), num_points_on_circle=20)


def format_score(score: int) -> str:
    if score > 0:
        return f"YELLOW +{score}"
    if score < 0:
        return f"RED +{abs(score)}"
    return "TIED"


def describe_stones(game: Curling) -> str:
    if not game.stones:
        return "No stones in play."

    ordered = sorted(
        game.stones,
        key=lambda stone: game.button_distance(stone),
    )
    lines = []
    for index, stone in enumerate(ordered, start=1):
        color = stone.color.name
        distance = float(game.button_distance(stone))
        lines.append(
            f"{index:>2}. {color:<6} pos=({stone.position[0]: .3f}, {stone.position[1]: .3f}) dist={distance:.3f}"
        )
    return "\n".join(lines)


@dataclass
class RandomAgent:
    rng: np.random.Generator
    name: str = "random"

    def select_shot(self, game: Curling, color: StoneColor) -> StoneThrow:
        del game
        means = StoneThrow.random_parameters[:, 0]
        stds = StoneThrow.random_parameters[:, 1]
        raw = self.rng.normal(means, stds)
        lower = StoneThrow.bounds[:, 0]
        upper = StoneThrow.bounds[:, 1]
        sqrt_velocity, angle, spin = np.clip(raw, lower, upper)
        return StoneThrow(color=color, sqrt_velocity=float(sqrt_velocity), angle=float(angle), spin=float(spin))


@dataclass
class HumanAgent:
    name: str = "human"

    def select_shot(self, game: Curling, color: StoneColor) -> StoneThrow:
        print()
        print(f"{color.name} to throw.")
        print(f"Current score: {format_score(game.evaluate_position())}")
        print(describe_stones(game))
        print("Enter: sqrt_velocity angle spin")
        while True:
            raw = input("> ").strip()
            try:
                sqrt_velocity_text, angle_text, spin_text = raw.split()
                sqrt_velocity = float(sqrt_velocity_text)
                angle = float(angle_text)
                spin = float(spin_text)
            except ValueError:
                print("Expected three numbers: sqrt_velocity angle spin")
                continue
            return StoneThrow(color=color, sqrt_velocity=sqrt_velocity, angle=angle, spin=spin)


def build_agent(name: str, rng: np.random.Generator) -> Agent:
    if name == "human":
        return HumanAgent()
    if name == "random":
        return RandomAgent(rng=rng)
    raise ValueError(f"Unsupported agent '{name}'")


def show_board(game: Curling):
    # Use a single rendered frame so the current board is visible while waiting for the next shot.
    game.display(SimulationConstants(time_intervals=0))


def play_end(
    red_agent: Agent,
    yellow_agent: Agent,
    starting_color: StoneColor | None,
    constants: SimulationConstants,
    display: bool,
):
    game = Curling(starting_color)
    for shot_number in range(1, game.num_stones_per_end + 1):
        if display:
            show_board(game)
        color = game.next_stone_color
        agent = red_agent if color == StoneColor.RED else yellow_agent
        shot = agent.select_shot(game, color)
        print()
        print(
            f"Shot {shot_number:>2}: {color.name} via {agent.name} "
            f"(sqrt_velocity={shot.sqrt_velocity:.3f}, angle={shot.angle:.3f}, spin={shot.spin:.3f})"
        )
        game.throw(shot, constants=constants, display=display)
        if display:
            show_board(game)
        print(f"Score after shot: {format_score(game.evaluate_position())}")

    print()
    print("Final board")
    print(describe_stones(game))
    print(f"Final score: {format_score(game.evaluate_position())}")
    if display:
        show_board(game)
        input("Press Enter to close the board...")


def main():
    parser = argparse.ArgumentParser(description="Play a curling end using the local simulator.")
    parser.add_argument("--red-agent", choices=("human", "random"), default="human")
    parser.add_argument("--yellow-agent", choices=("human", "random"), default="random")
    parser.add_argument("--starting-color", type=parse_color, default="random")
    parser.add_argument("--constants", choices=("fast", "accurate"), default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--display", action="store_true", help="Render throws with OpenCV windows.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    constants = build_constants(args.constants)
    play_end(
        red_agent=build_agent(args.red_agent, rng),
        yellow_agent=build_agent(args.yellow_agent, rng),
        starting_color=args.starting_color,
        constants=constants,
        display=args.display,
    )


if __name__ == "__main__":
    main()
