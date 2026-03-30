"""Play against the Orca bot or watch it play.

Usage:
    python examples/play_orca.py              # watch Orca vs heuristic
    python examples/play_orca.py --interactive  # play against Orca
"""
import sys; sys.path.insert(0, '..')
import argparse
from hexbot import Bot, Arena, HexGame

parser = argparse.ArgumentParser(description='Play with the Orca bot')
parser.add_argument('--interactive', action='store_true', help='Play against Orca')
parser.add_argument('--games', type=int, default=10, help='Games to watch (non-interactive)')
parser.add_argument('--sims', type=int, default=200, help='MCTS simulations per move')
args = parser.parse_args()

orca = Bot.orca(sims=args.sims)
print(f'Loaded: {orca}')

if args.interactive:
    print('\nYou are Player 0 (first move). Enter moves as "q r" (e.g. "0 0").')
    print('Type "quit" to exit.\n')
    game = HexGame()
    while not game.is_over:
        print(game)
        if game.current_player == 0:
            while True:
                try:
                    inp = input('Your move (q r): ').strip()
                    if inp.lower() == 'quit':
                        sys.exit(0)
                    q, r = map(int, inp.split())
                    game.place(q, r)
                    break
                except (ValueError, IndexError):
                    print('Invalid input. Enter two integers like "0 0"')
        else:
            move = orca.best_move(game)
            game.place(*move)
            print(f'Orca plays: {move}')
    print(game)
    print(f'Winner: Player {game.winner}')
else:
    print(f'\nOrca vs Heuristic ({args.games} games)')
    result = Arena(orca, Bot.heuristic(), num_games=args.games).play()
    print(f'\nOrca: {result.wins[0]}W  Heuristic: {result.wins[1]}W')
