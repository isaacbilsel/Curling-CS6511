import sys
import os
import numpy as np
from copy import deepcopy
import math
import random
from src.curling import Curling, SimulationConstants, StoneColor, StoneThrow

accurate_constants = SimulationConstants(time_intervals=.05)

def check_winner_state(state, color):
    score = state.evaluate_position()
    if color == StoneColor.YELLOW:
        return 1 if score < 0 else 0
    else:
        return 1 if score > 0 else 0

def available_actions():
    return [(sqrt_velocity, angle, spin)
        for sqrt_velocity in [1.35, 1.38, 1.40, 1.42]
        for angle in [-0.03, 0, 0.03]
        for spin in [1, 0, -1]
        ] 

def random_actions():
    return (np.random.uniform(1.35, 1.46),
            np.random.uniform(-.06, .05),
            np.random.uniform(2., -2.),)

class MCTSNode:
    def __init__(self, state, parent=None, action=None, player=None):
        self.state = state
        self.parent = parent
        self.action = action        
        self.player = player         
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = available_actions()

    def is_terminal(self):
        return self.state.get_state()['run_number'] >= 16
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        action = self.untried_actions.pop()
        new_state = deepcopy(self.state)

        player_to_move = new_state.next_stone_color
        new_state.throw(
            StoneThrow(
                color=player_to_move,
                sqrt_velocity=action[0],
                angle=action[1],
                spin=action[2],
            ),
            display=False, # false by default
            constants=accurate_constants # leave blank for the best trade off between speed and accuracy
        )

        child = MCTSNode(new_state, parent=self, action=action, player=player_to_move)
        self.children.append(child)
        return child
    
    def rollout(self, my_color):
        state = deepcopy(self.state)

        while state.get_state()['run_number'] < 16:
            player = state.next_stone_color
            move = random.choice(available_actions())
            state.throw(StoneThrow(
                    color=player,
                    sqrt_velocity=move[0],
                    angle=move[1],
                    spin=move[2],
                ))
        return check_winner_state(state, my_color)
    
    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1-result)
    
    def best_child(self, c=1.4):
        def ucb(child):
            if child.visits == 0:
                return float('inf')
            exploit = child.wins / child.visits
            explore = c * math.sqrt(math.log(self.visits) / child.visits)
            return exploit + explore
        return max(self.children, key=ucb)
    
def mcts_search(root_state, iterations=500):
    root = MCTSNode(root_state, player=StoneColor.YELLOW)

    for _ in range(iterations):
        node = root

        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        winner = node.rollout(my_color = StoneColor.YELLOW)
        node.backpropagate(winner)

    best = max(root.children, key=lambda c: c.visits)
    return best.action

def play():
    my_color = StoneColor.YELLOW
    curling = Curling(my_color)
    curling.reset(starting_color=my_color)
    states = []

    for i in range(curling.num_stones_per_end):
    # for i in range(2):
        current_player = curling.next_stone_color
        if current_player == StoneColor.YELLOW:
            move = mcts_search(deepcopy(curling), iterations=100)
            actions = StoneThrow(
                color=current_player,
                sqrt_velocity=move[0],
                angle=move[1],
                spin=move[2]
                )
        else:
            move = random_actions()
            actions = StoneThrow(
                color=current_player,
                sqrt_velocity=move[0],
                angle=move[1],
                spin=move[2]
                )

        curling.throw(actions,
            display=False,
            constants=accurate_constants
        )
        print(f"Move: {move}, Turn: {i}, Player: {current_player}, State: {curling.get_state()}")
    
    score = curling.evaluate_position()
    if score < 0:
        print('RED WINS')
    elif score > 0:
        print('YELLOW WINS')
    else:
        print('DRAW')

if __name__ == "__main__":
    play()