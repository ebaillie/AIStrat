# AIStrat
Python code for the El Grande game - standalone version

## INSTALLATION

1) Download the code to an appropriate place with python3 installed


2) Install OpenSpiel using the instructions copied from here:
https://openspiel.readthedocs.io/en/latest/install.html

NOTE: OpenSpiel instructions recommend the use of a virtual environment - instructions in the document. Virtual environment not currently in use, since this is a dedicated machine whose whole purpose is to run OpenSpiel code, so we’d never NOT want to be in the virtual environment - under the circumstances it seems non-useful. If venv is installed then when working on the code, start the virtual environment with
$ source venv/bin/activate
And leave it with
$ deactivate

NOTE2: OpenSpiel is under active development, so installing the latest may involve the need to tweak this code. It currently is compatible with the openSpiel code current at June 25 2021

Getting and building the code:

$ git clone https://github.com/deepmind/open_spiel.git

$ cd open_spiel

$ ./install.sh

$ ./open_spiel/scripts/build_and_run_tests.sh

Add these lines to ~/.profile (or other appropriate place where they will be read on startup)

#For the python modules in open_spiel.

export PYTHONPATH=$PYTHONPATH:/home/ubuntu/AIStrat/open_spiel

#For the Python bindings of Pyspiel

export PYTHONPATH=$PYTHONPATH:/home/ubuntu/AIStrat/open_spiel/build/python

(Edit path as/if necessary - I have always installed open_spiel as a subdirectory of this code but I imagine it's not necessary to do this)

## USE (in a python3 environment)

import el_grande

#Example Python code to start playing a 2-person game
#create a game

egame = el_grande.ElGrandeGame()

#starting state

estate = egame.new_initial_state()

#display starting state

estate

#display possible actions with string rep of which player and meaning of action

[(c,estate.action_to_string(c)) for c in estate.legal_actions()]

#first player plays Power Card 6

estate.apply_action(50)

#Copying and restarting games in the middle, for saving to db

import json

#get the state

estate._ json_for_game_state()

#initialise a new game from here

egame2=el_grande.ElGrandeGame({"game_state_json":json.dumps(estate._ json_for_game_state())})


#Parameterised game with different map and 4 players

egame=el_grande.ElGrandeGame({"players":4,"config_file":"el_grande_alta"})


