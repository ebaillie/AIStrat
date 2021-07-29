import copy
import pickle
import json

import numpy as np
import random
import couchdb
import pyspiel
import el_grande
import mcts_ext as mcts

#meta game currently structured to have 4 players, normal decks and scoreboards, variable regions

_DEFAULT_CONFIG = json.dumps({"players":["Red","White","Blue","Green"],
                "cardinfo":[7,2,4,1,8,11,3,9,6,10,5,2,6,4,8,5,3,1,7,9,11,6,9,4,2,3,10,8,7,5,1,2,4,10,8,7,9,5,6,3,11,1,1],
                "grandes":{"Red":"Granada","White":"Pais Vasco","Blue":"Sevilla","Green":"Galicia"},
                "king":"Castilla la Vieja",
                "config":"el_grande_default"})
_DEFAULT_SIMS = 10
_QUICKHEX = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D"]

_PLAYERS = 4
_DECKS = 5
_MAX_REGIONS = 12 # as current el_grande
_MAX_CAB_AREAS = _MAX_REGIONS + 3 # cab areas include province, court and castillo
_MAX_POWER_CARDS = 13 #as in original El Grande
_MAX_SCOREBOARDS = 2 #as in original El Grande game

#game phases power,action,end
_NUM_PHASES = 4 #for calculating possible max game size
_MAX_TURNS = 9 #full game has 9 turns, but we can specify fewer if need be

#phases in order
_PHASE_POWER=0
_PHASE_ACTION=1
_PHASE_END=2
_PHASE_NAMES = ['power','action','end']
_PHASE_IDS = {'power':_PHASE_POWER,'action':_PHASE_ACTION,'end':_PHASE_END}

#Power cards are 13 bitwise indicators (1 bit per player) for power and powerpast

#list of action numbers from 0 up

_ACT_POWERS = 0 #start of 'select a power card' actions
_ACT_DECKS = _ACT_POWERS + _MAX_POWER_CARDS #select deck
_ACT_END = _ACT_DECKS + _DECKS 

_ST_IDCH = 3 #channels for observer

_GAME_TYPE = pyspiel.GameType(
    short_name="meta_grande",
    long_name="Meta Grande",
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

class MetaGrandeGameState(pyspiel.State):
    """Meta Grande Game in open_spiel format
    """

    def __init__(self, game):
        super().__init__(game)
        self._game = game
        self._estate = self._game._egame.new_initial_state()
        self._deck_cards = np.where(self._estate._acard_round==self._estate._get_round())[0]
        self._history=[]

        rng=np.random.RandomState()
        evaluator=mcts.RandomRolloutEvaluator(1, rng)
        self._mbot = mcts.MCTSBot(self._game._egame,2,self._game._sims,evaluator,random_state=rng,solve=True,verbose=False)
    
    # OpenSpiel (PySpiel) API functions are below. These need to be provided by
    # every game. Some not-often-used methods have been omitted.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._estate._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        else:
            return self._estate._cur_player

    def legal_actions(self, player=None):
        """Returns a list of legal actions, sorted in ascending order.
        Args:
          player: the player whose legal moves
        Returns:
          A list of legal moves, where each move is in [0, num_distinct_actions - 1]
          at non-terminal states, and empty list at terminal states.
        """
        if player is not None and player != self._estate._cur_player:
            return []
        elif self.is_terminal():
            return []
        else:
            actions = []
            if self._estate._get_current_phase_name()=='power':
                cards = self._estate._available_powers()
                actions = [c+_ACT_POWERS for c in cards]
            else:
                #must be action - choices are decks 
                actions = [(i + _ACT_DECKS) for i in range (_DECKS) if self._deck_cards[i]>0] 
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
        if player is not None and player != self._estate._cur_player:
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
        #print("Player "+str(self._estate._cur_player)+": "+str(action))
        """Applies the specified action to the state"""

        #possible actions: _ACT_POWERS, _ACT_DECKS _ACT_CHOOSE_REGIONS, 

        #don't apply an illegal action
        if not action in self.legal_actions():
            return

        self._history += [action]

        if action>=_ACT_POWERS and action < _ACT_POWERS + _MAX_POWER_CARDS:
            self._estate.apply_action(action - _ACT_POWERS + el_grande._ACT_POWERS)
        else:
            #choose this deck
            cardid = self._deck_cards[action - _ACT_DECKS]
            this_player=self._estate._cur_player
            checkround=self._estate._get_round()
            
            #choose the card action then play out the card action including sensible simulation of likely results
            self._estate.apply_action(cardid + el_grande._ACT_CARDS)
            self._estate._end_turn=3*((self._estate._get_round()+2)//3)
            plan = self._mbot.multi_step(self._estate,'this_player')
            self._estate._end_turn=9
            for act in plan[1]:
                if self._estate._get_current_phase_name()[:6]=='action':
                    self._estate.apply_action(act)
            #if we have really low simulation power, this may not get to the end of action steps - just choose random
            while self._estate._get_current_phase_name()=='response' or (this_player==self._estate._cur_player and 
                                                                self._estate._get_current_phase_name()[:6]=='action'):
                self._estate.apply_action(random.choice(self._estate.legal_actions()))
            #skip scoring round if applicable
            while self._estate._get_current_phase_name()=='scoring' and not self._estate.is_terminal():
                self._estate.apply_action(random.choice(self._estate.legal_actions()))
            #mark deck as unavailable
            self._deck_cards[action - _ACT_DECKS]=-1
            #check if we need to refresh the decks, including ensuring deck 5 is set on
            if checkround!=self._estate._get_round():
                self._deck_cards=np.where(self._estate._acard_round==self._estate._get_round())[0]
                self._deck_cards=np.append(self._deck_cards,(len(self._estate._acard_round)-1))

    def action_to_string(self, arg0, arg1=None):
        """Action -> string. Args either (player, action) or (action)."""
        player = self.current_player() if arg1 is None else arg0
        action = arg0 if arg1 is None else arg1
        actionString=""
        if action >= _ACT_POWERS and action < _ACT_POWERS + _MAX_POWER_CARDS:
            actionString = "Power "+str(action + 1 - _ACT_POWERS)
        elif action>=_ACT_DECKS and action < _ACT_DECKS + _DECKS:
            cardid = self._deck_cards[action - _ACT_DECKS]
            actionString = "Deck {0} - {1}".format((action + 1 - _ACT_DECKS),self._estate._card_info(cardid)["name"])
        else:
            actionString = "Choose "+ self._estate_game._regions[action - _ACT_CHOOSE_REGION]
        
        return "{} ({})".format(self._estate._players[player],actionString)

    def is_terminal(self):
        return self._estate._is_terminal

    def returns(self):
        return self._estate._win_points

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
        return self._num_players

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
        # courtcabs/provincecabs  - 0..6 or M for Max
	# powercard played - 0 for none or card in hex
        retstr=str(self._get_round())+"/"
        for p in range(self._estate._num_players):
            cc=self._estate._region_cabcount(self._estate._game._court_idx,p)
            retstr+=str(cc) if cc<7 else "M"
        retstr+="/"
        for p in range(self._estate._num_players):
            cc=self._estate._region_cabcount(self._estate._game._province_idx,p)
            retstr+=str(cc) if cc<7 else "M"
        retstr+="/"
        for p in range(self._estate._num_players):
            pc = self._estate._get_power_card()
            cc=0 if len(pc)==0 else pc[0]+1
            retstr+=_QUICKHEX[cc]

        return retstr

    def clone(self):
        my_copy = self._game.new_initial_state()
        my_copy._estate = self._estate.clone()
        my_copy._deck_cards = self._deck_cards.copy() 

        return my_copy

_games=[]

class MetaGrandeGame(pyspiel.Game):
    """Meta Grande Game
    """

    def __init__(self,params={"config":None,"sims":None}):
        super().__init__(_GAME_TYPE, _GAME_INFO, params)
        _games.append(self)

        self._config = _DEFAULT_CONFIG
        self._sims = _DEFAULT_SIMS

        if params.get("config",None) is not None:
            self._config=params["config"]
        
        if params.get("sims",None) is not None:
            self._config=params["sims"]
        
        self._egame = el_grande.ElGrandeGame({"game_state_json":self._config})

    def new_initial_state(self):
        return MetaGrandeGameState(self)

    def num_distinct_actions(self):
        return _ACT_END

    def policy_tensor_shape(self):
        return (_ACT_END, 1)

    def clone(self):
        return MetaGrandeGame()

    def max_chance_outcomes(self):
        return 0

    def get_parameters(self):
        return {}

    def num_players(self):
        return self._egame._num_players

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
        return MetaGrandeGameObserver()


class MetaGrandeGameObserver:
    """Observer, conforming to the PyObserver interface (see observer.py).
       3 bits per player - court/province 0..6 or 7+
       4 bits per player - power card currently played (0 for unplayed)
       4 bits total - round number
    """
    
    def __init__(self):
        tensor_size = _PLAYERS*(2*_ST_IDCH + 4) + 4
        self.tensor = np.zeros(tensor_size, np.float32)
        self._court = self.tensor[:_PLAYERS*_ST_IDCH]
        self._province = self.tensor[_PLAYERS*_ST_IDCH:2*_PLAYERS*_ST_IDCH]
        self._power = self.tensor[2*_PLAYERS*_ST_IDCH:-4]
        self._round = self.tensor[-4:]
        self.dict = {"court": self._court,"province":self._province, "power": self._power, "round":self._round}


    def set_from(self, state, player):
        del player
        court=np.array([max(2^_ST_IDCH,c) for c in state._estate._board_state[state._estate._game._court_idx,:_PLAYERS]])
        province=np.array([max(2^_ST_IDCH,p) for p in state._estate._board_state[state._estate._game._province_idx,:_PLAYERS]])
        for ch in range(_ST_IDCH):
            cmat=(court>>ch)%2
            pmat=(province>>ch)%2
            self._court[ch*_PLAYERS:(ch+1)*_PLAYERS]=cmat
            self._province[ch*_PLAYERS:(ch+1)*_PLAYERS]=pmat

        binstr=""
        for p in range(state._estate._num_players):
            pc = state._estate._get_power_card(p)
            cc = 0 if len(pc)==0 else pc[0]+1
            binstr+=bin(32+cc)[4:]
        self._power[:]=np.array([int(b) for b in binstr])
        self._round[:]=np.array([int(b) for b in bin(32+state._estate._get_round())[4:]])

    def string_from(self, state, player):
        del player
        return str(state)

pyspiel.register_game(_GAME_TYPE,MetaGrandeGame)
