import copy
import pickle
import json

import numpy as np
import random
import couchdb
import pyspiel

#simple game currently structured to have 4 players, normal decks, scoreboards and regions

_QUICKHEX = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D"]
_COURTCAT = [0,1,2,3,4,4,5,5,5,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]
_PROVCAT = [0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

_PLAYERS = 4
_DECKS = 5
_POWER_CARDS = 13 #as in original El Grande
_SCOREBOARDS = 2 #as in original El Grande game

#game phases power,action,end
_NUM_PHASES = 3 #for calculating possible max game size
_MAX_TURNS = 9 

#phases in order
_PHASE_POWER=0
_PHASE_ACTION=1
_PHASE_END=2
_PHASE_NAMES = ['power','action','end']
_PHASE_IDS = {'power':_PHASE_POWER,'action':_PHASE_ACTION,'end':_PHASE_END}

#Power cards are 13 bitwise indicators (1 bit per player) for power and powerpast

#list of action numbers from 0 up

_ACT_POWERS = 0 #start of 'select a power card' actions
_ACT_DECKS = _ACT_POWERS + _POWER_CARDS #select deck
_ACT_END = _ACT_DECKS + _DECKS 

_PROVINCE=0
_COURT=1
_REGION=2

_GAME_TYPE = pyspiel.GameType(
    short_name="simple_grande",
    long_name="Simple Grande",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION, 
    utility=pyspiel.GameType.Utility.GENERAL_SUM, 
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_PLAYERS,
    min_num_players=_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=True,
    provides_observation_string=False,
    provides_observation_tensor=False,
    provides_factored_observation_string=False)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_ACT_END,  
    max_chance_outcomes=0,  
    num_players=_PLAYERS,
    min_utility=0.0,
    max_utility=1.0,
    utility_sum=1.0,
    max_game_length=_PLAYERS*_MAX_TURNS*_NUM_PHASES) 

class SimpleGrandeGameState(pyspiel.State):
    """ Simple Grande Game in open_spiel format

    """

    def __init__(self, game):
        super().__init__(game)
        self._game = game
        self._cabs = np.array([21,7,2,21,7,2,21,7,2,21,7,2]).reshape(_PLAYERS,3)
        self._score = np.full(_PLAYERS,0.0)
        self._win_points = np.full(_PLAYERS,0.0)
        self._is_terminal=False
        self._deck_available = np.full(_DECKS,True)
        self._power_available = np.full((_PLAYERS,_POWER_CARDS),True)
        self._power_available_now = np.full(_POWER_CARDS,True)
        self._power_out=[-1,-1,-1,-1]
        self._round=1
        self._phase=_PHASE_POWER 
        self._history=[]
        self._cur_player=0
    
    # OpenSpiel (PySpiel) API functions are below. These need to be provided by
    # every game. Some not-often-used methods have been omitted.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        else:
            return self._cur_player

    def legal_actions(self, player=None):
        """Returns a list of legal actions, sorted in ascending order.
        Args:
          player: the player whose legal moves
        Returns:
          A list of legal moves, where each move is in [0, num_distinct_actions - 1]
          at non-terminal states, and empty list at terminal states.
        """
        if player is not None and player != self.current_player():
            return []
        elif self.is_terminal():
            return []
        else:
            actions = []
            if self._phase==_PHASE_POWER:
                avail = np.where(self._power_available[self._cur_player,:])[0]
                avail_now = np.where(self._power_available_now)[0]
                actions = (_ACT_POWERS + np.intersect1d(avail,avail_now)).tolist()
            else:
                #must be action - choices are decks 
                actions =  (_ACT_DECKS + np.where(self._deck_available)[0]).tolist()
            assert(len(actions)>0)
            return actions
    
    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        return []

    def legal_actions_mask(self, player=None):
        """Get a list of legal actions.
        Args:
          player: the player whose moves we want; defaults to the current player.
        Returns:
          A list of legal moves, where each move is in [0, num_distinct_actios - 1].
          Returns an empty list at terminal states, or if it is not the specified
          player's turn.
        """
        if player is not None and player != self._cur_player:
            return []
        elif self.is_terminal():
            return []
        else:
            action_mask = [0] * _ACT_END
            for action in self.legal_actions():
                action_mask[action] = 1
            return action_mask


    def apply_action(self, action):
        self.do_apply_action(action)

    def do_apply_action(self, action):
        #print("Player "+str(self._cur_player)+": "+str(action))
        """Applies the specified action to the state"""

        #possible actions: _ACT_POWERS, _ACT_DECKS  

        #don't apply an illegal action
        if not action in self.legal_actions():
            return

        self._history += [action]

        if action>=_ACT_POWERS and action < _ACT_POWERS + _POWER_CARDS:
            self._power_out[self._cur_player]=(action - _ACT_POWERS)
            self._power_available_now[action - _ACT_POWERS]=False
            self._power_available[self._cur_player,(action - _ACT_POWERS)]=False
            self._set_round_phase()
        else:
            #choose this deck
            self._deck_available[action - _ACT_DECKS]=False
            self._cab_movement(self._cur_player,(action - _ACT_DECKS + 1))
            self._score[self._cur_player]+=self._score_card(self._cur_player,(action - _ACT_DECKS + 1))
            self._set_round_phase()

    def _cab_movement(self,player,deck):
        #power card determines province -> court
        prov_to_court=min(self._cabs[player,_PROVINCE],(6 - (self._power_out[player]+1)//2))
        self._cabs[player,_PROVINCE]-=prov_to_court
        self._cabs[player,_COURT]+=prov_to_court
        court_to_region = min(deck,self._cabs[player,_COURT])
        self._cabs[player,_COURT]-=court_to_region
        self._cabs[player,_REGION]+=court_to_region

    def _score_card(self, player, deck):
        #use heuristics - boost for Deck 3 if you have high cabs out, boost for Deck 1 if you have low cards
        #boost for deck 5 if it's round 3/6/9
        score = 0
        if deck==5 and self._round%3 == 0:
            score += 2
        elif deck==3:
            score += 4*(self._cabs[player,_REGION]/max(self._cabs[:,_REGION]))
        elif deck==1:
            score += 1-(self._cabs[player,_REGION]/max(self._cabs[:,_REGION]))
        #score boosts count for more at higher rounds
        score = score * self._round
        return score
            
    def _set_round_phase(self):
        #set current player
        if self._phase==_PHASE_POWER:
            self._cur_player = (self._cur_player + 1) % _PLAYERS 
            if self._power_out[self._cur_player]>-1:
                self._phase=_PHASE_ACTION
                self._cur_player=self._power_out.index(max(self._power_out))
        else:
            powersleft=[a for a in self._power_out if a<self._power_out[self._cur_player]]
            if len(powersleft)==0:
                if self._round % 3 == 0:
                    newscores = 5 * (self._cabs[:,_REGION]/max(self._cabs[:,_REGION]))
                    self._score += newscores
 
                if self._round==_MAX_TURNS:
                    self._is_terminal=True
                    self._win_points = self._score//max(self._score)
                else:
                    self._round+=1
                    self._phase=_PHASE_POWER
                    self._cur_player=self._power_out.index(min(self._power_out)) 
                    self._power_out=[-1,-1,-1,-1]
                    self._power_available_now=np.full(_POWER_CARDS,True)
                    self._deck_available=np.full(_DECKS,True)
            else:
                self._cur_player=self._power_out.index(max(powersleft))

 
    def action_to_string(self, arg0, arg1=None):
        """Action -> string. Args either (player, action) or (action)."""
        player = self.current_player() if arg1 is None else arg0
        action = arg0 if arg1 is None else arg1
        actionString=""
        if action >= _ACT_POWERS and action < _ACT_POWERS + _POWER_CARDS:
            actionString = "Power "+str(action + 1 - _ACT_POWERS)
        elif action>=_ACT_DECKS and action < _ACT_DECKS + _DECKS:
            actionString = "Deck {0}".format((action + 1 - _ACT_DECKS))
        
        return "{} ({})".format(player,actionString)

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        return self._win_points

    def rewards(self):
        return self.returns()

    def player_reward(self, player):
        return self.rewards()[player]

    def player_return(self, player):
        return self.returns()[player]

    def is_chance_node(self):
        return False

    def is_simultaneous_node(self):
        return False

    def history(self):
        return self._history

    def history_str(self):
        return str(self._history)

    def child(self, action):
        cloned_state = self.clone()
        cloned_state.apply_action(action)
        return cloned_state

    def apply_actions(self, actions):
        raise NotImplementedError  # Only applies to simultaneous move games

    def num_distinct_actions(self):
        return _ACT_END

    def num_players(self):
        return _PLAYERS

    def get_game(self):
        return self._game

    def get_type(self):
        return self._game.get_type()

    def serialize(self):
        return pickle.dumps(self)

    def resample_from_infostate(self):
        return [self.clone()]

    def __str__(self):
	# round number
        # courtcabs/provincecab  categories
	# powercard played - 0 for none or card in hex
        retstr=str(self._round)+"/"
        for p in range(_PLAYERS):
            cc=self._cabs[p,_COURT]
            retstr+=str(_COURTCAT[cc]) 
        retstr+="/"
        for p in range(_PLAYERS):
            cc=self._cabs[p,_PROVINCE]
            retstr+=str(_PROVCAT[cc]) 
        retstr+="/"
        for p in range(_PLAYERS):
            retstr+=_QUICKHEX[self._power_out[p]+1]

        return retstr

    def clone(self):
        my_copy = self._game.new_initial_state()
        my_copy._deck_cards = self._deck_cards.copy() 
        my_copy._cabs = self._cabs.copy()
        my_copy._score = self._score.copy()
        my_copy._win_points = self._win_points.copy()
        my_copy._is_terminal = self._is_terminal
        my_copy._deck_available = self._deck_available.copy()
        my_copy._power_available = self._power_available.copy()
        my_copy._power_available_now = self._power_available_now.copy()
        my_copy._power_out = self._power_out.copy()
        my_copy._round = self._round
        my_copy._phase = self._phase 
        my_copy._history = self._history.copy()
        my_copy._cur_player = self._cur_player

        return my_copy

_games=[]

class SimpleGrandeGame(pyspiel.Game):
    """Simple Grande Game
    """

    def __init__(self,params={}):
        super().__init__(_GAME_TYPE, _GAME_INFO, params)
        _games.append(self)

    def new_initial_state(self):
        return SimpleGrandeGameState(self)

    def num_distinct_actions(self):
        return _ACT_END

    def policy_tensor_shape(self):
        return (_ACT_END, 1)

    def clone(self):
        return SimpleGrandeGame()

    def max_chance_outcomes(self):
        return 0

    def get_parameters(self):
        return {}

    def num_players(self):
        return _PLAYERS

    def min_utility(self):
        return 0.0

    def max_utility(self):
        return 1.0

    def get_type(self):
        return _GAME_TYPE

    def utility_sum(self):
        return 0.0

    def observation_tensor_layout(self):
        return pyspiel.TensorLayout.CHW

    def deserialize_state(self, string):
        return pickle.loads(string)

    def __str__(self):
        return "python_meta_grande_game"

    def make_py_observer(self, iig_obs_type, params):
        if params:
            raise ValueError("Params not supported")
        return SimpleGrandeGameObserver()


class SimpleGrandeGameObserver:
    """Observer, conforming to the PyObserver interface (see observer.py).
       3 bits per player - court 0,1,2-3,4-6,7-11,12+
       2 bits per player - province 0-1,2-5,6-11,12+
       5 bits total - power cards currently played in buckets: [0-1,2-4,5-7,8-10,11-12] 
       2 bits total - round number/3
    """
    
    def __init__(self):
        tensor_size = _PLAYERS*(3 + 2 ) + 7 + 2 
        self.tensor = np.zeros(tensor_size, np.float32)
        self._court = self.tensor[:_PLAYERS*3]
        self._province = self.tensor[_PLAYERS*3:_PLAYERS*5]
        self._power = self.tensor[_PLAYERS*5:-2]
        self._round = self.tensor[-2:]
        self.dict = {"court": self._court,"province":self._province, "power": self._power, "round":self._round}


    def set_from(self, state, player):
        del player
        court=np.array([_COURTCAT[c] for c in state._cabs[:,_COURT]])
        province=np.array([_PROVCAT[p] for p in state._cabs[:,_PROVINCE]])
        for ch in range(3):
            cmat=(court>>ch)%2
            self._court[ch*_PLAYERS:(ch+1)*_PLAYERS]=cmat
        for ch in range(2):
            pmat=(province>>ch)%2
            self._province[ch*_PLAYERS:(ch+1)*_PLAYERS]=pmat

        self._power[0]=1 if any(state._power_available_now[0:2]) else 0 
        self._power[1]=1 if any(state._power_available_now[2:5]) else 0
        self._power[2]=1 if any(state._power_available_now[5:8]) else 0
        self._power[3]=1 if any(state._power_available_now[8:11]) else 0
        self._power[4]=1 if any(state._power_available_now[11:13]) else 0
        self._round[:]=np.array([int(b) for b in bin(8+(state._round%3))[4:]])

    def string_from(self, state, player):
        del player
        return str(state)

pyspiel.register_game(_GAME_TYPE,SimpleGrandeGame)
