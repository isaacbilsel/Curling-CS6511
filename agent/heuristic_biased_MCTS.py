from __future__ import annotations
from src.curling import Curling, SimulationConstants, Stone, StoneColor, StoneThrow
#from numba import jit
import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
from copy import deepcopy


class MCTS_Tree_Config:
    """
      Iterations number of MCTS
      number of actions of each dimension in [sqrt_velocity, angle, spin]    
      number of children nodes 
      two strategies to choose actions:
            1. Randomly sample in a continuous action space with a small number of heuristic near center pitching actions.
            2. Discretize the continuous action space into a fixed grid and sample & search within the grid actions.
      ......
      ......

    """
    def __init__(self, iterations: int,  exploration_weight: float = 1.44, 
                random_seed:Optional[int] = None, 
                children_number: int = 16,
                action_per_dimension: int=None):
                
        self.iterations = iterations
        self.children_number = children_number
        self.action_per_dimension = action_per_dimension
        self.accurate_constants = SimulationConstants(time_intervals=0.2, num_points_on_circle=10)
        self.fast_constants = SimulationConstants(time_intervals=1.0, num_points_on_circle=4)  # lower precision for rollout
        self.randomseed = random_seed
        self.C = exploration_weight


class MCTS_Tree_Node:
    def __init__(self, game_state: Curling, player_color: StoneColor,
                parent_node: Optional["MCTS_Tree_Node"] = None,
                parent_throw: Optional[Tuple[float, float, float]] = None,
                children_node_number: int = None,
                action_per_dimension: int = None,
                actions: List[Tuple[float, float, float]] = None,
                ):
        """
           config:
              Curling game state
              father node
              father action
              color
              children node list
              sum_value
              sum_visits
              unvisited actions list
              ......
              ......
        """
        self.game_state = game_state
        self.player_color = player_color
        self.parent_node = parent_node
        self.parent_throw = parent_throw
        self.children_list : List[MCTS_Tree_Node] = []
        self.children_node_number = children_node_number
        self.action_per_dimension = action_per_dimension
        self.sum_value = 0
        self.sum_visits = 0
        if actions is not None and not is_the_last_round(game_state):
            self.unvisited_actions = actions
        else:
            self.unvisited_actions = []

    def node_average_value(self):
        if self.sum_visits == 0:
            return 0
        return self.sum_value / self.sum_visits
    

    def exist_unvisited_node_actions(self):
        " Whether there are unvisited actions in A(s)"
        if self.unvisited_actions:
            return True
        else:
            return False
    
    def choose_unvisited_node_action(self, constant: SimulationConstants, actions: List[Tuple[float, float, float]])->"MCTS_Tree_Node":
        " "
        select_action = self.unvisited_actions.pop()
        new_game = copy_game(self.game_state)
        execute_a_throw(new_game, select_action, constant)
        child_node = MCTS_Tree_Node(new_game, self.player_color, parent_node=self, parent_throw=select_action, 
                                    children_node_number=self.children_node_number, action_per_dimension=self.action_per_dimension,
                                    actions=actions)
        self.children_list.append(child_node)
        return child_node
           
    def choose_the_best_action(self, C: float)->"MCTS_Tree_Node":
        """
        Use UCT: 
        The player is maximizing the value: value as much as possible, 
        visits as much as possible, 
        the opponent is minimizing the value: value as little as possible, 
        visits as much as possible
        """
        def uct(node: MCTS_Tree_Node):
           if self.game_state.next_stone_color == self.player_color:
                if node.sum_visits == 0:
                    return float('inf')
                return node.node_average_value() + C * math.sqrt(math.log(self.sum_visits) / node.sum_visits)
           else:
                if node.sum_visits == 0:
                    return float('-inf')
                return node.node_average_value() - C * math.sqrt(math.log(self.sum_visits) / node.sum_visits)

        if self.game_state.next_stone_color == self.player_color:
            best_child = max(self.children_list, key=uct)
        else:
            best_child = min(self.children_list, key=uct)
        return best_child


class MCTS_Agent:
    def __init__(self, config:MCTS_Tree_Config):
        self.config = config
        self.grid_actions = self.build_actions_grid()
        self.random_seed = random.Random(config.randomseed)
        
    def build_actions_grid(self):
        if self.config.action_per_dimension is None:
            return None

        limit = StoneThrow.bounds
        actions = []
        velocity_dis = np.linspace(limit[0][0], limit[0][1], self.config.action_per_dimension)
        angle_dis = np.linspace(limit[1][0], limit[1][1], self.config.action_per_dimension)
        spin_dis = np.linspace(limit[2][0], limit[2][1], self.config.action_per_dimension)
        for v in velocity_dis:
            for a in angle_dis:
                for s in spin_dis:
                    actions.append((float(v),float(a),float(s)))
        return actions 
            
    def build_no_grid_actions(self):
        if self.random_seed.random() <= 0.5:
            "sample round the center"
            Action = (self.random_seed.gauss(1.41, 0.02),self.random_seed.gauss(0.0, 0.01),self.random_seed.gauss(0.0, 0.04))
            limit = StoneThrow.bounds
            return (float(max(limit[0][0], min(limit[0][1], Action[0]))),
                    float(max(limit[1][0], min(limit[1][1], Action[1]))),
                    float(max(limit[2][0], min(limit[2][1], Action[2]))))
        else:
            limit = StoneThrow.bounds
            return (self.random_seed.uniform(limit[0][0], limit[0][1]), 
                    self.random_seed.uniform(limit[1][0], limit[1][1]), 
                    self.random_seed.uniform(limit[2][0], limit[2][1]))

    "Real Actions in this game"
    def sample_real_action(self):
        if self.grid_actions is None:
            actions = []
            for i in range(self.config.children_number):
                actions.append(self.build_no_grid_actions())
            return actions
        else:
            actions = list(self.grid_actions)
            self.random_seed.shuffle(actions)
            return actions

    def rollout(self, game_state: Curling, color:StoneColor):
        """ Play until the end (provided that the node is unvisited)."""
        current_game = copy_game(game_state)
        throw_number = 0
        while not is_the_last_round(current_game):
            if self.grid_actions is None:
                throw_action = self.build_no_grid_actions()
            else:
                throw_action = self.random_seed.choice(self.grid_actions)
            execute_a_throw(current_game, throw_action, self.config.fast_constants)
            throw_number += 1
        return calculate_score(current_game, color)

    def search(self, curling:Curling):
        "Iteration"
        if is_the_last_round(curling):
            raise RuntimeError("This is the last round!")
        player_color = curling.next_stone_color
        root_node = MCTS_Tree_Node(copy_game(curling), player_color, parent_node=None, parent_throw = None,
                                    children_node_number = self.config.children_number, 
                                    action_per_dimension = self.config.action_per_dimension,
                                    actions = self.sample_real_action())
        for i in range(self.config.iterations):
            node = self.find_an_unvisited_node(root_node)
            score = self.rollout(node.game_state, node.player_color)
            self.back_propagate(node, score)
        if root_node.children_list:
            best_child = max(root_node.children_list, key=lambda x: (x.sum_visits, x.sum_value))
            if best_child.parent_throw is not None:
                action= best_child.parent_throw
            else:
                action = self.build_no_grid_actions()
        else:
            action = self.build_no_grid_actions()

        return StoneThrow(player_color, action[0], action[1], action[2])

    def back_propagate(self, node:MCTS_Tree_Node, score:float):
        while node is not None:
            node.sum_visits += 1
            node.sum_value += score
            node = node.parent_node
        
    def find_an_unvisited_node(self, node:MCTS_Tree_Node):
        root = node
        while not is_the_last_round(root.game_state):
            if root.exist_unvisited_node_actions():
                result_node = root.choose_unvisited_node_action(self.config.accurate_constants, self.sample_real_action())
                return result_node
            if not root.children_list:
                return root
            root = root.choose_the_best_action(self.config.C)
        return root

    def get_action(self, curling:Curling):
        return self.search(curling)


def is_the_last_round(curling:Curling):
    a = curling.get_state()["run_number"]
    b = curling.num_stones_per_end
    return a >= b

def execute_a_throw(curling:Curling, throw_action:Tuple[float, float, float], constant:SimulationConstants):
    limit = StoneThrow.bounds 
    action = (np.clip(throw_action[0], limit[0][0], limit[0][1]),
              np.clip(throw_action[1], limit[1][0], limit[1][1]),
              np.clip(throw_action[2], limit[2][0], limit[2][1]))
    curling.throw(StoneThrow(curling.next_stone_color, action[0], action[1], action[2]), constant, display = False)

def copy_game(curling:Curling):
    return deepcopy(curling)

def calculate_score(curling:Curling, player_color:str):
    return float(curling.evaluate_position()) * float(player_color.value)


def play_a_turn(mcts_agent:MCTS_Agent, curling:Curling):
    throw = mcts_agent.search(curling)
    print(throw)
    curling.throw(throw, mcts_agent.config.accurate_constants, display = False)
    return curling




# MCTS Agent vs MCTS Agent
def play_a_game_with_MCTS(mcts_agent:MCTS_Agent):
    curling = Curling(StoneColor.RED)
    curling.reset(starting_color=StoneColor.RED)
    for i in range(curling.num_stones_per_end):
        if is_the_last_round(curling):    
            break
        print("The number of stones throw is", i+1)
        curling = play_a_turn(mcts_agent, curling)
        print(curling.get_state())
    return curling
  
accurate_constants = SimulationConstants(time_intervals=0.2, num_points_on_circle=10)

# Heuristic Agent vs MCTS Agent
def play_a_game_vs_heuristic(mcts_agent:MCTS_Agent):
    curling = Curling(StoneColor.RED)
    for i in range(curling.num_stones_per_end):
        if is_the_last_round(curling):
            break
        if curling.next_stone_color == StoneColor.RED:
            print("The number of stones throw is", i+1)
            curling = play_a_turn(mcts_agent, curling)
            print(curling.get_state())
        else:
            # Heuristic: throw towards center with minimal angle/spin
            curling.throw(
                StoneThrow(color=StoneColor.YELLOW, sqrt_velocity=1.41, angle=np.random.uniform(-0.01, 0.01), spin=np.random.uniform(-0.1, 0.1)),
                constants=accurate_constants, display=False,
            )
    return curling

# Random Agent vs MCTS Agent
def play_a_game_vs_random(mcts_agent:MCTS_Agent):
    curling = Curling(StoneColor.RED)
    for i in range(curling.num_stones_per_end):
        if is_the_last_round(curling):
            break
        if curling.next_stone_color == StoneColor.RED:
            print("The number of stones throw is", i+1)
            curling = play_a_turn(mcts_agent, curling)
            print(curling.get_state())
        else:
            # Random: throw with random parameters
            curling.throw(
                StoneThrow(color=StoneColor.YELLOW, sqrt_velocity=np.random.uniform(1.35, 1.46), angle=np.random.uniform(-0.06, 0.05), spin=np.random.uniform(-2.0, 2.0),),
                constants=accurate_constants,
                display=False,
            )
    return curling

mctsagent = MCTS_Agent(
    MCTS_Tree_Config(
        iterations=210,
        children_number=36,
        random_seed=5
    )
)
curling = play_a_game_vs_heuristic(mctsagent)
print("Final Score: ", curling.evaluate_position())