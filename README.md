#### CSCI 6511: AI Algorithms 
#### Curling Strategy Agent

## Problem and Motivation

The game of curling involves a combination of strategy and technique. Finding an optimal strategy is nontrivial as it requires thinking well in advance and considering all the uncertainty (adversary’s moves and error in technique). Using AI Algorithms to find optimal strategies can assist curling teams in their preparation and training.

## State Space:

The state space is the locations of the agent's and opponent's team's stones on a 2D grid. We represent this as a dictionary containing 'stones' as list of (x, y, color) tuples and 'run_number' as int. See the get_state() function in curling.py.

```
State = {
"stones": [(x₁, y₁, c₁), (x₂, y₂, c₂), ..., (xₙ, yₙ, cₙ)],
"run_number": t
}
```

`(x, y)` = position of stone on the 2D sheet
`c` = color of stone
- +1 = Yellow
- -1 = Red
`t` = turn_number
<!-- 
```
{'stones': [(np.float64(0.04853583313857754), np.float64(-5.512993542856897), <StoneColor.RED: -1>), (np.float64(1.64446281434387), np.float64(-5.556416746422605), <StoneColor.YELLOW: 1>), (np.float64(-0.173198472030246), np.float64(-5.317171842047518), <StoneColor.RED: -1>), (np.float64(-1.3676833474923775), np.float64(-6.2549993437489295), <StoneColor.YELLOW: 1>), (np.float64(0.7806616385920546), np.float64(-7.364115150758466), <StoneColor.RED: -1>), (np.float64(0.3529680538083919), np.float64(-6.275979374595629), <StoneColor.YELLOW: 1>), (np.float64(0.34944742854327504), np.float64(-5.376133875953437), <StoneColor.RED: -1>), (np.float64(-1.0421147017595493), np.float64(-5.152593784730243), <StoneColor.YELLOW: 1>), (np.float64(0.23626325570568849), np.float64(-5.813596980909742), <StoneColor.RED: -1>), (np.float64(-0.7682910757868106), np.float64(-5.861123698241833), <StoneColor.YELLOW: 1>), (np.float64(-0.40850738280796106), np.float64(-5.714268998188931), <StoneColor.RED: -1>), (np.float64(1.963073359750351), np.float64(-5.607813777493406), <StoneColor.YELLOW: 1>), (np.float64(0.3188093829264815), np.float64(-7.174160532212673), <StoneColor.RED: -1>), (np.float64(-0.4609753075170058), np.float64(-8.911978763291675), <StoneColor.YELLOW: 1>), (np.float64(-0.5385741457766272), np.float64(-6.784967210907723), <StoneColor.RED: -1>), (np.float64(1.194854431344018), np.float64(-6.465060064351937), <StoneColor.YELLOW: 1>)], 'run_number': 16}
``` -->
## Action Space

The action space is the throw parameters: angle, velocity, spin, each of which are real numbers. Each above variable has an uncertainty/throw error e.
## Agent

We use a Monte Carlo Tree Search (MCTS) to determine the optimal action based on the game state. See agent/heuristic_biased_MCTS.py. The agent searches over throw parameters `(sqrt_velocity, angle, spin)` and returns a `StoneThrow` for the current player. Each MCTS iteration runs: Selection, Expansion, Rollout, and Backpropagation. Selection uses the standard upper confidence bound tree (UCT) formula:
$$UCT_i = \bar{X}_i + C \sqrt{\frac{\ln N}{n_i}}$$
where $\bar{X}_i$ is the average reward/value of child node i, $n_i$ is the number of visits to child node i,
$N$ is the number of visits to the parent node, and 
$C$ is the exploration constant. We perform rollout using heuristic opponent actions. 

We utilize two strategies to choose actions during the selection step:
1. Grid Mode: Randomly sample in a continuous action space with a small number of heuristic near center pitching actions.
2. No-grid Mode: Discretize the continuous action space into a fixed grid and sample & search within the grid actions.

It should be noted that:
When there is no need to expand for a node and thus we are searching for the best child, two situations are considered in the function `choose_the_best_action`
1. When the next player is the root player, maximize the UCT, that is $$UCT_i = \bar{X}_i + C \sqrt{\frac{\ln N}{n_i}}$$
2. When the next player is an island, in order to create trouble for the rootplayer, the UCT is minimized, that is $$UCT_i = \bar{X}_i + C \sqrt{\frac{\ln N}{n_i}}$$

## Testing
We test our agent agains a random agent, which chooses legal actions uniform randomly, and a heuristic agent, which always throws the stone exactly into the center ring. These agents are implemented in agent/simpleAgents.py. 

## Results

Run to test:
`python3 -m agent.simpleAgents`

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