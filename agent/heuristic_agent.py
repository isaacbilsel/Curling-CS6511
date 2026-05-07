from src.curling import Curling, SimulationConstants, StoneColor, StoneThrow

accurate_constants = SimulationConstants(time_intervals=0.05)


def get_state(curling):
    stone_states = []
    for stone in curling.stones:
        stone_states.append((stone.position, stone.color))
    return stone_states


def closest_in_house(curling):
    stones_in_house = []
    for stone in curling.stones:
        if curling.in_house(stone):
            stones_in_house.append(stone)
    if len(stones_in_house) == 0:
        return None
    return min(stones_in_house, key=curling.button_distance)


def heuristic_agent(curling, my_color):
    shot_stone = closest_in_house(curling)

    if shot_stone is None:
        return StoneThrow(
            color=my_color,
            sqrt_velocity=1.41,
            angle=0.0,
            spin=0.0,
        )

    if shot_stone.color != my_color:
        angle = 0.0
        if shot_stone.position[0] < -0.2:
            angle = 0.02
        elif shot_stone.position[0] > 0.2:
            angle = -0.02
        return StoneThrow(
            color=my_color,
            sqrt_velocity=1.58,
            angle=angle,
            spin=0.0,
        )

    angle = 0.025
    spin = 1.0
    if shot_stone.position[0] > 0:
        angle = -0.025
        spin = -1.0
    return StoneThrow(
        color=my_color,
        sqrt_velocity=1.34,
        angle=angle,
        spin=spin,
    )


if __name__ == "__main__":
    my_color = StoneColor.YELLOW
    other_color = StoneColor.RED
    curling = Curling(my_color)
    curling.reset(starting_color=my_color)
    states = []

    for i in range(curling.num_stones_per_end):
        if curling.next_stone_color == my_color:
            action = heuristic_agent(curling, my_color)
        else:
            action = heuristic_agent(curling, other_color)

        curling.throw(
            action,
            display=False,
            constants=accurate_constants,
        )

        state = get_state(curling)
        states.append(state)

    score = curling.evaluate_position()
    print(score)
    print(states)
