"""Tabular Q-Learner example on SimpleGrande, focusing on final player position.

Q-Learning agents are trained by playing against each other, random, and mcts. Then, the game
can be played against the agents from the command line.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
from absl import app
from absl import flags
import numpy as np
import simple_grande
import pickle
import collections
import mcts_ext as mcts
import random

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(5e4), "Number of train episodes.")
flags.DEFINE_boolean(
    "interactive_play", True,
    "Whether to run an interactive play with the agent after training.")

logging.basicConfig(filename='sg_qlearner.log', level=logging.INFO)
NUM_PLAYERS = 4

def command_line_action(time_step):
  """Gets a valid action from the user on the command line."""
  current_player = time_step.observations["current_player"]
  legal_actions = time_step.observations["legal_actions"][current_player]
  action = -1
  while action not in legal_actions:
    print("Choose an action from {}:".format(legal_actions))
    sys.stdout.flush()
    action_str = input()
    try:
      action = int(action_str)
    except ValueError:
      continue
  return action


def eval_against_random_bots(env, pos, trained_agents, random_agents, mbot, num_episodes):
  """Evaluates trained agent at `pos` against two `random_agents` and an mcts bot for `num_episodes`."""
  wins = 0
  cur_agents = random_agents
  mbot_id=random.choice([i for i in range(NUM_PLAYERS) if i!=pos])
  cur_agents[pos]=trained_agents[pos]
  for _ in range(num_episodes):
    time_step = env.reset()
    gameState=mbot._game.new_initial_state()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      if player_id==mbot_id:
        action=mbot.step(gameState)
      else:
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        action=agent_output.action
      time_step = env.step([action])
      gameState.apply_action(action)
    if time_step.rewards[pos] > 0:
      wins += 1
  return wins / num_episodes


def main(_):
  gamename = "simple_grande"
  num_players =4 
  key_agent_id = 3
  sims=100

  env = rl_environment.Environment(gamename)
  num_actions = env.action_spec()["num_actions"]
  game = simple_grande.SimpleGrandeGame()
  rng = np.random.RandomState()
  evaluator = mcts.RandomRolloutEvaluator(1, rng)
  mbot = mcts.MCTSBot(game,2,sims,evaluator,random_state=rng,solve=True,verbose=False)
  
  agents = [
      tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  # random agents for evaluation
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  # 1. Train the agents
  training_episodes = FLAGS.num_episodes
  for cur_episode in range(training_episodes):
    if cur_episode % int(1e3) == 0:
      win_rates = eval_against_random_bots(env, key_agent_id, agents, random_agents, mbot, 100)
      logging.info("Starting episode %s, win_rates %s", cur_episode, win_rates)
    time_step = env.reset()
    gameState = mbot._game.new_initial_state()
    training_agents=[agents[i] for i in range(num_players)]
    #occasionally put in a random agent or an mcts agent
    rand=random.randint(0,10)
    mbot_id=-1
    if rand<3:
      training_agents[rand]=random_agents[rand]
    elif rand==10:
      mbot_id=random.randint(0,2)
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      if player_id==mbot_id:
        action=mbot.step(gameState)
      else:
        agent_output = training_agents[player_id].step(time_step)
        action=agent_output.action
      time_step = env.step([action])
      gameState.apply_action(action)

    # Episode is over, step all agents with final info state.
    for agent in training_agents:
      if agent._player_id!=mbot_id:
        agent.step(time_step)

  dumpname='sg_qtab_agent3.pickle'
  pickle.dump( agents[key_agent_id], open( dumpname, "wb" ) )
  
  if not FLAGS.interactive_play:
    return

  # 2. Play from the command line against the trained agent.
  human_player =0 
  if True:
    print("You are playing as {0}".format(human_player))
    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      if player_id == human_player:
        agent_out = agents[human_player].step(time_step, is_evaluation=True)
        #logging.info("\n%s", agent_out.probs)
        #logging.info("\n%s", pretty_board(time_step))
        print("\n%s", env._state)
        print("\n%s", [(c,env._state.action_to_string(c)) for c in env._state.legal_actions()])
        action = command_line_action(time_step)
      else:
        agent_out = agents[player_id].step(time_step, is_evaluation=True)
        action = agent_out.action
      time_step = env.step([action])

    logging.info("\n%s", time_step)

    logging.info("End of game!")
    if time_step.rewards[human_player] > 0:
      logging.info("You win")
    elif time_step.rewards[human_player] < 0:
      logging.info("You lose")
    else:
      logging.info("Draw")
    # Switch order of players
    human_player = 1 - human_player


if __name__ == "__main__":
  app.run(main)
