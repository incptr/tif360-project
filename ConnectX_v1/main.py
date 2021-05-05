from agents import *
from obs import Obs

n_games = 10
# agent1 = None
# agent1 = N_steps_look_ahead_agent()
agent1 = AB_agent()
agent2 = AB_agent()
# agent2 = N_steps_look_ahead_agent()
agents = [agent1, agent2]
obs = Obs(rows=6, cols=7, x=4, agents=agents)
obs.play_n_games(n_games)