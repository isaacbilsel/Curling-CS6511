import numpy as np
from copy import deepcopy
import math
import random
from src.curling import Curling, SimulationConstants, StoneColor, StoneThrow
import pandas as pd

accurate_constants = SimulationConstants(time_intervals=.2)

def check_winner_state(state, color):
    score = state.evaluate_position()
    return (score+8)/16

def available_actions():
    return [(sqrt_velocity, angle, spin)
        for sqrt_velocity in [1.38, 1.39, 1.40, 1.41, 1.42]
        for angle in [-0.02, 0, 0.02]
        for spin in [1, 0, -1]
        ] 

def random_actions():
    return (np.random.normal(1.40, 0.02),
            np.random.normal(0, 0.02),
            np.random.normal(0, 0.5))

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
            if player == my_color:
                move = random.choice(available_actions())
                actions = StoneThrow(
                    color=player,
                    sqrt_velocity=move[0],
                    angle=move[1],
                    spin=move[2]
                    )
            else:
                move = random_actions()
                actions = StoneThrow(
                    color=player,
                    sqrt_velocity=move[0],
                    angle=move[1],
                    spin=move[2]
                    )

            state.throw(actions,
                        display=False,
                        constants=accurate_constants
                        )
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

def throw_error(action):
    sqrt_velocity, angle, spin = action
    return (np.clip(sqrt_velocity + np.random.normal(0, 0.03), 1.36, 1.46),
    np.clip(angle + np.random.normal(0, 0.01), -0.1, 0.1),
    np.clip(spin + np.random.normal(0, 0.5), -4, 4))

def play():
    my_color = StoneColor.YELLOW
    curling = Curling(my_color)
    curling.reset(starting_color=my_color)
    log = []

    for i in range(curling.num_stones_per_end):
        current_player = curling.next_stone_color
        if current_player == StoneColor.YELLOW:
            move = mcts_search(deepcopy(curling), iterations=500)
            move_error = throw_error(move)
            actions = StoneThrow(
                color=current_player,
                sqrt_velocity=move_error[0],
                angle=move_error[1],
                spin=move_error[2]
                )
        else:
            move = random_actions()
            move_error = throw_error(move)
            actions = StoneThrow(
                color=current_player,
                sqrt_velocity=move_error[0],
                angle=move_error[1],
                spin=move_error[2]
                )

        curling.throw(actions,
                      display=False,
                      constants=accurate_constants
                      )
        
        latest = curling.stones[-1] if curling.stones else None
        x = float(latest.position[0]) if latest else None
        y = float(latest.position[1]) if latest else None

        log.append({
            'turn': i,
            'player': current_player,
            'actual': move,
            'errored': move_error,
            'x': x,
            'y': y
        })
        print(f'Turn: {i+1}')
    
    score = curling.evaluate_position()
    winner = 'YELLOW' if score > 0 else 'RED' if score < 0 else 'DRAW'
    return log, score, winner

def loop(n=10):
    alllogs = []
    wins, losses, draws = 0, 0, 0
    for i in range(n):
        print(f'Game: {i+1}')
        log, score, winner = play()
        for j in log:
            j['game'] = i + 1
            j['winner'] = winner
            j['final_score'] = score
        alllogs.extend(log)

        if score > 0:
            wins += 1
        elif score < 0:
            losses += 1
        else:
            draws += 1

        print(f'Game: {i+1}, Score: {score}, Wins: {wins}, Losses: {losses}, Draws: {draws}')
        
    df = pd.DataFrame(alllogs)
    df.to_csv('results.csv', index=False)

    print(f'Wins: {wins/n*100:.1f}%')
    print('DONE')

if __name__ == "__main__":
    loop(n=5)