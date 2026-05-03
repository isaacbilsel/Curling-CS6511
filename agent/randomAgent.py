from src.curling import Curling, SimulationConstants, StoneColor, StoneThrow
import numpy as np

accurate_constants = SimulationConstants(time_intervals=.05)

# def get_state(curling):
#     stone_states = []
#     for i in curling.stones:
#         stone_states.append((i.position, i.color))
#     return stone_states

def max_agent(curling):
    return StoneThrow(
            color=my_color,
            sqrt_velocity=np.random.uniform(1.35, 1.46),
            angle=np.random.uniform(-.06, .05),
            spin=np.random.uniform(2., -2.),
        )

def min_agent(curling):
    return StoneThrow(
            color=curling.next_stone_color,
            sqrt_velocity=np.random.uniform(1.35, 1.46),
            angle=np.random.uniform(-.06, .05),
            spin=np.random.uniform(2., -2.),
        )

my_color = StoneColor.YELLOW
curling = Curling(my_color)
curling.reset(starting_color=my_color)
states = []


for i in range(4):
    if curling.next_stone_color == my_color:
        action = max_agent(curling)
    else:
        action = min_agent(curling)

    curling.throw(action,
        display=True,
        constants=accurate_constants
    )

    state = curling.get_state()
    states.append(state)

score = curling.evaluate_position()
print(score)
print(states)