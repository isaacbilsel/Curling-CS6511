# CSCI 6511: AI Algorithms 
## Curling Strategy Agent

### How to run:
1. Clone repositoty git@github.com:isaacbilsel/Curling-CS6511.git
2. Install dependencies: pip install -r requirements.txt
3. In the last line of the mctsAgent.py, specify how many games you want to run
4. To run a single game, call loop(n=1) and in the terminal run 
`python3 -m agent.mctsAgent`
5. Results will be saved to results.csv

Note: Due to deepcopy of the state at every turn and the physics simulator, each game takes 25-30 minutes to run based on the iterations.
You can also change the time intervals between the simulation at the top of the script. Higher the simulation constant, faster the agent simulator is, but recommended .2 for better results, as faster simulation can discrupt the physics

### Problem Statement 
The game of curling involves a combination of strategy and technique. Finding an optimal strategy is nontrivial as it requires thinking well in advance and considering all the uncertainty (adversary’s moves and error in technique). Using AI Algorithms to find optimal strategies can assist curling teams in their preparation and training.

### Related solutions
a. The below repo is our source for a curling physics simulator. The author used this simulator to create a MCST + NN agent called betacurl but it was
unfinished/unsuccessful.
https://github.com/George-Ogden/curling
b. The below competition involves a curling simulator with a PPO agent
implementation, however the problem formulation is not representative of real curling.
https://github.com/jidiai/Competition_Olympics-Curling/tree/main
c. The below paper outlines a methodology for modeling curling as a markov process.
https://edwards.usask.ca/faculty/Keith%20Willoughby/files/EJOR%202001.pdf
d. The below ICML paper from 2018 uses a deep CNN and a monte-carlo search tree based on a C++ physics simulator.
https://proceedings.mlr.press/v80/lee18b/lee18b.pdf
e. The below work (2025) uses an actor-critic algorithm to assess curling strategy.
https://www.research-collection.ethz.ch/server/api/core/bitstreams/59f77456-9ead-4381-a745-936402e7bdc7/content

#### State Space
The state space is the locations of the agent's and opponent's team's stones on a 2D grid. We represent this as a dictionary containing 'stones' as list of (x, y, color) tuples and 'run_number' as int. See the `get_state()` function in curling.py.

```
State = {
"stones": [(x₁, y₁, c₁), (x₂, y₂, c₂), ..., (xₙ, yₙ, cₙ)],
"run_number": t
}
```

`(x, y)` = position of stone on the 2D sheet `c` = color of stone

- +1 = Yellow
- -1 = Red `t` = turn_number

#### Action space
The action space is the throw parameters: angle, velocity, spin, each of which are real numbers. Each above variable has an uncertainty/throw error e.

#### Transitions
The transitions are determined by the physics simulator. It includes trajectory, curl, and collisions. Depends on the actions and error
`S' = f(S, A+e)`

Gaussian noise (error) is added to the action selected by the agent. This was done to model the fact that humans cannot execute the throws perfectly. Error is applied to each parameter independently. See `throw_error` function in mctsAgent.py

#### Observation
The state space is fully observable. The agent can see or have information about the complete board, stones and their positions before start of every turn.

### Solution method
We have applied Monte Carlo Tree Search to choose actions. MCTS is choosen becuase of the large state space and random nature of the environment.

Each time the agent takes a turn, it run 500 iterations of the four steps in MCTS algorithms - Selection, Expansion, Rollout, and Backpropagation

Starting from the root node, the algorithm tries to find an untried action with best UCB. Once the untried action is selected from the available combinations, the simulator executes the thrwo and the resulting state is represented as child node. From this child node, the rest of the game is simulated randomly to completion. The MCTS agent will pick from its discretized action space, while the opponent picks from the random distribution. The simulated result is backpropagated back throught the tree. The reward fucntion uses a normalized score `(score+8)/16`. This converts the simulator score into a value between 0 and 1. After 500 iterations, the action with most visits is selected as the best move

### Implementation
We implemented a complete MCTS agent that plays curling against a random opponent. The project was implemented in Python3 using the curling simulator forked from George-Ogden/curling. This simulator models real curling dynamics including friction, curl, and collisions. We have built an MCTS agent on top of this simulator. 
Key components:
- MCTSNode: stores the game state, parent node, action taken, visit count, win value and list of untried actions
- mcts_search(): run iteration of MCTS loop (Selection -> Expansion -> Rollout -> Backpropagation)
- available_actions(): returns 45 discrete action combinations

```
sqrt_velocity : [1.38, 1.39, 1.40, 1.41, 1.42]
angle : [-0.02, 0, 0.02]
spin : [1, 0, -1]
```

- random actions(): sampling for opponent using normal distribution matching the simulator parameters
- throw_error(): adds guassian noise to simulate human execution error
- play(): runs one full game, all 16 turns between the agent and opponent

#### Software and Hardware requirements
- Python3
- numpy, pandas, opencv
- Any standard laptop is enough

Each game takes 25 minutes at 500 iterations due to the computational cost of deepcopy at every turn and the physics simulation

### Evaluation:
The agent was tested by playing games against a random agent. The agent has win rate of 40% against the random agent. All the runs were logged and results were part of `results.csv`  file.

### References used:
Simulator: https://github.com/George-Ogden/curling
For outline of the code: https://www.geeksforgeeks.org/machine-learning/monte-carlo-tree-search-mcts-in-machine-learning/




## Original Repo:

# Curling
Simulated curling environment  
![Rendered Curling Environment](docs/images/environment.png)  
Physics based on [Dynamics and curl ratio of a curling stone](https://rdcu.be/dgIW2)  
Used in [https://github.com/George-Ogden/betacurl](https://github.com/George-Ogden/betacurl)
## Install
With pip
```sh
pip install git+https://github.com/George-Ogden/curling.git
```
from source
```sh
git clone https://github.com/George-Ogden/curling
pip install .
```
## Usage
```python
from curling import Curling, SimulationConstants, StoneColor, StoneThrow
import numpy as np

accurate_constants = SimulationConstants(time_intervals=.05)

# leave blank for a random starting stone
current_player = StoneColor.RED
curling = Curling(current_player)
curling.reset(starting_color=current_player) # optional for first game

for i in range(curling.num_stones_per_end):
    current_player = curling.next_stone_color
    # throw a stone
    curling.throw(
        StoneThrow(
            color=current_player,
            # sqrt velocity is specified
            sqrt_velocity=np.random.uniform(1.35, 1.46),
            angle=np.random.uniform(-.06, .05),
            spin=np.random.uniform(2., -2.),
        ),
        display=True, # false by default
        constants=accurate_constants # leave blank for the best trade off between speed and accuracy
    )

print(curling.evaluate_position()) # positive for YELLOW and negative for RED
```
## Documentation
For more information, see the documentation at [https://curling.readthedocs.io/](https://curling.readthedocs.io/)