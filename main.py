from agents import *
from Game import Game

n_games = 1000

# choosing agents
path = 'strat_v4.pt'
agent1 = DQN_Agent(load_path=path)
agent2 = One_step_ahead()
agents = [agent1, agent2]

# instantiating game, and running for n_games
game = Game(rows=6, cols=7, inarow=4, agents=agents)
game.play_n_games(n_games)