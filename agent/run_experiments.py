from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time

from src.curling import Curling, SimulationConstants, StoneThrow
from src.enums import StoneColor
from agent.mctsHeuristicRollout import mcts, sim_constants
from agent.heuristic_agent import heuristic_agent

def random_throw(color):
    return StoneThrow(
        color=color,
        sqrt_velocity=np.random.uniform(1.35, 1.46),
        angle=np.random.uniform(-0.06, 0.05),
        spin=np.random.uniform(-2.0, 2.0),
    )

def play_game(args):
    game_num, opponent = args
    print(f"[vs {opponent}] Game {game_num}: starting...", flush=True)
    start = time.time()
    game = Curling(StoneColor.RED)

    for i in range(game.num_stones_per_end):
        color = game.next_stone_color
        if color == StoneColor.RED:
            throw = mcts(game)
        elif opponent == "random":
            throw = random_throw(color)
        elif opponent == "heuristic":
            throw = heuristic_agent(game, color)
        game.throw(throw, constants=sim_constants, display=False)
        print(f"[vs {opponent}] Game {game_num}: turn {i+1}/16 done", flush=True)

    score = game.evaluate_position()
    elapsed = time.time() - start
    red_stones = sum(1 for s in game.stones if s.color == StoneColor.RED)
    yellow_stones = sum(1 for s in game.stones if s.color == StoneColor.YELLOW)
    result = "RED wins" if score < 0 else "YELLOW wins" if score > 0 else "Draw"
    print(f"[vs {opponent}] Game {game_num}: DONE in {elapsed:.1f}s — score={score} → {result} (RED stones={red_stones}, YELLOW stones={yellow_stones})", flush=True)
    return (opponent, score)

if __name__ == "__main__":
    tasks = [(i, "random") for i in range(1, 51)] + [(i, "heuristic") for i in range(1, 51)]
    print(f"Starting 100 games in parallel (50 vs random, 50 vs heuristic)...\n", flush=True)
    total_start = time.time()

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(play_game, tasks))

    print(f"\nAll done in {time.time()-total_start:.1f}s\n")
    for opponent in ["random", "heuristic"]:
        scores = [s for opp, s in results if opp == opponent]
        red_wins = sum(1 for s in scores if s < 0)
        yellow_wins = sum(1 for s in scores if s > 0)
        draws = sum(1 for s in scores if s == 0)
        print(f"vs {opponent}:")
        total = len(scores)
        print(f"  RED  (MCTS):       {red_wins}/{total}")
        print(f"  YELLOW (opponent): {yellow_wins}/{total}")
        print(f"  Draws:             {draws}/{total}")
        print()
