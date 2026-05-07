import math
import random
from copy import deepcopy
import numpy as np

from src.curling import Curling, SimulationConstants, StoneThrow
from src.enums import StoneColor

ITERATIONS = 50
C = 1.4
ROLLOUT_DEPTH = 4

sim_constants = SimulationConstants(time_intervals=0.2)
fast_constants = SimulationConstants(time_intervals=1.0)


def heuristic_throw(game, col):
    stones_in = [s for s in game.stones if game.in_house(s)]
    if not stones_in:
        return StoneThrow(color=col, sqrt_velocity=1.41, angle=0.0, spin=0.0)

    closest = min(stones_in, key=game.button_distance)

    if closest.color != col:
        ang = 0.02 if closest.position[0] < 0 else -0.02
        return StoneThrow(color=col, sqrt_velocity=1.58, angle=ang, spin=0.0)
    else:
        ang = 0.025 if closest.position[0] < 0 else -0.025
        return StoneThrow(color=col, sqrt_velocity=1.35, angle=ang, spin=0.5)


def sample_actions(n=20):
    acts = []
    bounds = StoneThrow.bounds
    for _ in range(n):
        r = random.random()
        if r < 0.5:
            v = np.clip(random.gauss(1.41, 0.02), bounds[0][0], bounds[0][1])
            a = np.clip(random.gauss(0.0, 0.01), bounds[1][0], bounds[1][1])
            s = np.clip(random.gauss(0.0, 0.3), bounds[2][0], bounds[2][1])
        elif r < 0.75:
            v = np.clip(random.gauss(1.58, 0.02), bounds[0][0], bounds[0][1])
            a = np.clip(random.gauss(0.0, 0.02), bounds[1][0], bounds[1][1])
            s = np.clip(random.gauss(0.0, 0.2), bounds[2][0], bounds[2][1])
        else:
            v = random.uniform(bounds[0][0], bounds[0][1])
            a = random.uniform(bounds[1][0], bounds[1][1])
            s = random.uniform(bounds[2][0], bounds[2][1])
        acts.append((float(v), float(a), float(s)))
    return acts


class Node:
    def __init__(self, game, parent=None, action=None):
        self.game = game
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.val = 0.0
        self.untried = sample_actions(20)

    def is_terminal(self):
        return self.game.get_state()["run_number"] >= self.game.num_stones_per_end

    def is_expanded(self):
        return len(self.untried) == 0

    def ucb_score(self, child):
        if child.visits == 0:
            return float("inf")
        return (child.val / child.visits) + C * math.sqrt(math.log(self.visits) / child.visits)

    def best_child(self):
        return max(self.children, key=lambda c: self.ucb_score(c))

    def expand(self):
        act = self.untried.pop()
        new_game = deepcopy(self.game)
        new_game.throw(
            StoneThrow(new_game.next_stone_color, act[0], act[1], act[2]),
            constants=sim_constants,
            display=False,
        )
        child = Node(new_game, parent=self, action=act)
        self.children.append(child)
        return child


def rollout(game, my_col):
    sim = deepcopy(game)
    t = 0
    while sim.get_state()["run_number"] < sim.num_stones_per_end and t < ROLLOUT_DEPTH:
        throw = heuristic_throw(sim, sim.next_stone_color)
        sim.throw(throw, constants=fast_constants, display=False)
        t += 1
    return float(sim.evaluate_position()) * float(my_col)


def mcts(game):
    my_col = game.next_stone_color
    root = Node(deepcopy(game))

    for _ in range(ITERATIONS):
        node = root

        while not node.is_terminal() and node.is_expanded():
            node = node.best_child()

        if not node.is_terminal() and not node.is_expanded():
            node = node.expand()

        score = rollout(node.game, my_col)

        while node is not None:
            node.visits += 1
            node.val += score
            node = node.parent

    if not root.children:
        return heuristic_throw(game, my_col)

    best = max(root.children, key=lambda c: c.visits)
    return StoneThrow(my_col, best.action[0], best.action[1], best.action[2])


def random_throw(color):
    return StoneThrow(
        color=color,
        sqrt_velocity=np.random.uniform(1.35, 1.46),
        angle=np.random.uniform(-0.06, 0.05),
        spin=np.random.uniform(-2.0, 2.0),
    )

if __name__ == "__main__":
    game = Curling(StoneColor.RED)

    for i in range(game.num_stones_per_end):
        color = game.next_stone_color
        if color == StoneColor.RED:
            throw = mcts(game)
        else:
            throw = random_throw(color)
        game.throw(throw, constants=sim_constants, display=False)
        print(f"Turn {i+1} ({color.name}): {game.evaluate_position()}")

    score = game.evaluate_position()
    print("Final score:", score)
    if score < 0:
        print("RED wins")
    elif score > 0:
        print("YELLOW wins")
    else:
        print("Draw")
