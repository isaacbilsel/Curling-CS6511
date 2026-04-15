# Create a heuristic agent and test against a random agent
# Red = heueristic agent, yellow = random agent

from curling import Curling, SimulationConstants, StoneColor, StoneThrow
import numpy as np

accurate_constants = SimulationConstants(time_intervals=.2)

# leave blank for a random starting stone
current_player = StoneColor.RED
curling = Curling(current_player)
curling.reset(starting_color=current_player) # optional for first game

# Always throw the stone towards the center with almost no spin or angle. 
def heuristicAgent(color):
    curling.throw(
        StoneThrow(
            # Action space: 
            color=color,
            sqrt_velocity= 1.41, # np.random.uniform(1.40, 1.41),
            angle= np.random.uniform(-.01, .01),
            spin=np.random.uniform(0.1, -0.1),
        ),
        display=False, # false by default
        constants=accurate_constants # leave blank for the best trade off between speed and accuracy
    )

def randomAgent(color):
    curling.throw(
        StoneThrow(
            # Action Space: 
            color=color,
            sqrt_velocity=np.random.uniform(1.35, 1.46),
            angle=np.random.uniform(-.06, .05),
            spin=np.random.uniform(2., -2.),
        ),
        display=False, # false by default
        constants=accurate_constants # leave blank for the best trade off between speed and accuracy
    )

for i in range(curling.num_stones_per_end):
    current_player = curling.next_stone_color
    if current_player == StoneColor.RED:
        heuristicAgent(current_player)
    else: 
        randomAgent(current_player)

print("State: \n", curling.get_state())

print("\nFinal Score: ", curling.evaluate_position()) # positive for YELLOW and negative for RED