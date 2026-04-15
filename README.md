#### CSCI 6511: AI Algorithms 
#### Curling Strategy Agent

## Problem
The game of curling involves a combination of strategy and technique. Finding an optimal strategy is nontrivial as it requires thinking well in advance and considering all the uncertainty (adversary’s moves and error in technique). Using AI Algorithms to find optimal strategies can assist curling teams in their preparation and training. 

## State Space: 
The state space is the locations of the agent's and opponent's team's stones on a 2D grid. We represent this as a dictionary containing 'stones' as list of (x, y, color) tuples and 'run_number' as int. See the get_state() function in curling.py. 

## Action Space
The action space is the throw parameters: angle, velocity, spin, each of which are real numbers. Each above variable has an uncertainty/throw error e. 

## Agent 
A simple heuristic and random agent have been implemented to understand and test the state/action space descriptions. This is implemented in agent/simpleAgents.py.  




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