from Agents import *
from Game import Game

n_games = 20

# choosing agents
path = 'strat_v8.pt'
agent1 = DQN_Agent(load_path=path)
agent2 = One_step_ahead()

# instantiating game, and running for n_games
agents = [agent1, agent2]
game = Game(rows=6, cols=7, inarow=4, agents=agents)
scores = game.play_n_games(n_games//2)

agents = [agent2, agent1]
game.agents = agents
x = game.play_n_games(n_games//2)
scores.extend([[a, b] for [b, a] in x])

print_summary(scores, dqn=True)
