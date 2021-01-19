import copy
import pickle
import json

import numpy as np

import pyspiel

_MAX_PLAYERS = 5
_NUM_REGIONS = 9
_NUM_EXT_REGIONS = _NUM_REGIONS + 1 #regions including castillo
_NUM_CAB_AREAS = _NUM_EXT_REGIONS + 2 #areas caballeros can be placed, including court and province
#region order "Aragon","Castilla la Nueva","Castilla la Vieja","Cataluna","Galicia","Granada","Pais Vasco","Sevilla","Valencia",("Castillo","court","province")
_NUM_FULL_DECKS = 4 #decks with multiple cards in them
_NUM_PLAYABLE_DECKS = 5 #include also the single-card 'Deck5'
_MAX_DECK_COUNT = 11 #greatest number of cards in any single deck
_NUM_ACTION_CARDS = 43 #total number of action cards in all decks
_NUM_POWER_CARDS = 13
#game phases start,power,action,actionchoice,actioncard,actioncabs,score
_NUM_PHASES = 7
_PHASE_NAMES = ['start','power','action','actionchoose','actioncard','actioncab','score']
_MAX_TURNS = 9 #full game has 9 turns, but we can specify fewer if need be


_DEFAULT_PLAYERS = 4


#State matrix indicators
#Store state as a matrix of ints, so as to make it easier to convert to a tensor later

_ST_IDX_REGIONS = 0 #start of regions
_ST_IDX_CASTILLO = _ST_IDX_REGIONS + 1
_ST_IDX_COURT = _ST_IDX_CASTILLO + 1
_ST_IDX_PROVINCE = _ST_IDX_COURT + 1
_ST_IDX_POWERS = _ST_IDX_PROVINCE + 1 #start of current power cards
_ST_IDX_POWERPASTS = _ST_IDX_POWERS + _MAX_PLAYERS
_ST_IDX_DECKS = _ST_IDX_POWERPASTS + _MAX_PLAYERS
_ST_IDX_DECKPASTS = _ST_IDX_DECKS + _NUM_FULL_DECKS
_ST_IDX_GAMECONTROL = _ST_IDX_DECKPASTS + _NUM_FULL_DECKS #round, phase and score info and player order info
_ST_IDX_END = _ST_IDX_GAMECONTROL + 1

_ST_IDY_CABS = 0 #start of cab counts, in region columns
_ST_IDY_GRANDES = _ST_IDX_CABS + _MAX_PLAYERS #start of grande locations, in region columns
_ST_IDY_KING = _ST_IDX_GRANDES + _MAX_PLAYERS
_ST_IDY_SECRETSELS = _ST_IDY_KING + 1

_ST_IDY_ROUND = 0 #start of gamecontrol column - roundcount
_ST_IDY_PHASE = 1
_ST_IDY_SCORES = 2 #scores of player1,player2,...
_ST_IDY_PLAYER_QUEUE = _ST_IDY_SCORES + _MAX_PLAYERS
_ST_IDY_ACT_DONE = _ST_IDY_PLAYER_QUEUE + _MAX_PLAYERS #was the card action taken for the current player
_ST_IDY_CAB_DONE = _ST_IDY_ACT_DONE + 1 #were caballeros moved for the current player

_ST_IDY_END = max((_ST_IDY_SECRETSELS+_MAX_PLAYERS),(_ST_IDY_CAB_DONE+1),_NUM_POWER_CARDS,_MAX_DECK_COUNT) #current max should be 16

_ST_IDCH = 5 #highest number in the matrix should be 32, so want 5 channels

#list of action numbers from 0 up

_ACT_DEAL = 0
_ACT_CARDS = 1 #start of 'select a card' actions
_ACT_POWERS = _ACT_CARDS + _NUM_ACTION_CARDS #select power cards
_ACT_RETRIEVE_POWERS = _ACT_POWERS + _NUM_POWERS #get back an old power card
_ACT_DECIDE_CAB = _ACT_RETRIEVE_POWERS + _NUM_POWERS #decide to place cabs first
_ACT_DECIDE_ACT = _ACT_DECIDE_CAB + 1 #decide to play the action card first
_ACT_CHOOSE_SECRETS = _ACT_DECIDE_ACT+1 #choose one secret region
_ACT_MOVE_GRANDES = _ACT_CHOOSE_SECRETS + _NUM_REGIONS
_ACT_MOVE_KINGS = _ACT_MOVE_GRANDES + _NUM_REGIONS
_ACT_CAB_MOVES = _ACT_MOVE_KINGS + _NUM_REGIONS
_ACT_END = _ACT_CAB_MOVES + (_NUM_CAB_AREAS * _NUM_CAB_AREAS * _MAX_PLAYERS) #combos of moving a cab from region, to region, of player

class ElGrandeGameState(object):
    """El Grande Game in open_spiel format
    """

    def __init__(self, game):
        self._game = game
        self._cur_player = 0
        self._num_players = 0
        self._is_terminal = False
        self._history = []
        self._win_points = np.full(players, 0)
        self._winner = False
        self._state_matrix = np.full((_ST_IDX_END,_ST_IDY_END),0)
        self._cards = [] #cards will be loaded at state initialisation
        self._decks = {1:[],2:[],3:[],4:[],5:[]} #same cards, organised by deck
        self._regions = []
        self._players = []
        self._end_turn = _MAX_TURNS #default end turn is 9
        
    # Helper functions (not part of the OpenSpiel API).
    
    #info about card names and abilities, region names, player colours
    
    def _get_rid(self,regionName):
        return self._regions.index(regionName)
    
    def _get_pid(self,playerName):
        return self._players.index(playerName)

    def _get_cid(self,cardName):
        return self._cards.index(cardName)

    def _get_phaseid(self,phaseName):
        return _PHASE_NAMES.index(phaseName)

    def _load_game_info(self,jsonData):
        self._regions = jsonData["Regions"]
        self._players = jsonData["Players"]
        self._num_players = len(self._players)
        self._cards = jsonData["Cards"]
        assert(len(self._cards)==_NUM_ACTION_CARDS)
        #cards all named 'Deckn.cardname' - separate into decks for easier processing later
        for key in self._cards.keys():
            cardname = self._cards[key]["name"]
            decknumber = int(cardname[4])
            self._decks[decknumber].append(key)
            
    #turn all relevant state info from DB format into game format
    def _load_game_state(self,jsonData):
        self._state_matrix_add_king(jsonData['king'])
        self._state_matrix_add_cabs_grandes(jsonData['pieces'])
        self._state_matrix_add_turn_info(jsonData['turninfo'])
        #TODO - action cards done, available, currently with player
        
    def _state_matrix_add_king(self,region_name):
        region_id = self._get_rid(region_name)
        assert(region_id < _ST_IDX_CASTILLO )
        self._state_matrix[(_ST_IDX_REGIONS+region_id),_ST_IDY_KING]=1
        
    def _state_matrix_add_cabs_grandes(self,data):
        for player_name in data.keys():
            player_id = self._get_pid(player_name)
            for key in data[player_name].keys():
                if key=="grande":
                    region_id = self._get_region(data[player_name][key])
                    assert(region_id < _ST_IDX_CASTILLO)
                    self._state_matrix[(_ST_IDX_REGIONS+region_id),_ST_IDY_GRANDES + player_id]=1
                else:
                    region_id = self._get_region(key)
                    assert(region_id < _ST_IDX_POWERS)
                    self._state_matrix[(_ST_IDX_REGIONS+region_id),_ST_IDY_CABS + player_id]=data[player_name][key]

    def _state_matrix_add_turn_info(self,data):
        #power cards
        for player_name in data['powercards'].keys()
            player_id = self._get_pid(player_name)
            power_id = data['powercards'][player_name]
            assert(power_id <= _NUM_POWER_CARDS and power_id > 0) #power_id from 1 to _NUM_POWER_CARDS
            self._state_matrix[(_ST_IDX_POWERS+player_id),(power_id-1)]=1
        #past power cards
        for player_name in data['powerplayed'].keys()
            player_id = self._get_pid(player_name)
            for power_id in data['powercards'][player_name]:
                assert(power_id <= _NUM_POWER_CARDS and power_id > 0) #power_id from 1 to _NUM_POWER_CARDS
                self._state_matrix[(_ST_IDX_POWERPASTS+player_id),(power_id-1)]=1
        #round
        self._state_matrix[_ST_IDX_GAMECONTROL,_ST_IDY_ROUND]=data['round']
        #phase
        self._state_matrix[_ST_IDX_GAMECONTROL,_ST_IDY_PHASE]=self._get_phaseid(data['phase'])
        #score for each player
        for player_name in data['scores'].keys()
            player_id = self._get_pid(player_name)
            score = data['scores'][player_name]
            self._state_matrix[_ST_IDX_GAMECONTROL,_ST_IDY_SCORES+player_id]=score
        #queue
        player_queue = data['playersleft'][1:] #skip the first element, since that's the current player
        for k in range(len(player_queue)):
            self._state_matrix[_ST_IDX_GAMECONTROL,(_ST_IDY_PLAYER_QUEUE+k)]=self._get_pid(player_queue[k])
        #action and/or cabs for current player
        if len(data['playersleft'])>0:
            self._state_matrix[_ST_IDX_GAMECONTROL,_ST_IDY_ACT_DONE]=int(data['actionsdone'][data['playersleft'][0]])
            self._state_matrix[_ST_IDX_GAMECONTROL,_ST_IDY_CAB_DONE]=int(data['cabsdone'][data['playersleft'][0]])
                  
    #FROM CASTILLO GAME - AUDIT
    def _extract_region_counts(self,region):
        assert(region>=0 and region<=_NUM_REGIONS)
        region_substr = self._board[(region*self._num_players):((region+1)*self._num_players)]
        counts = [ord(s) - ord(_ZERO_CHAR) for s in region_substr]
        return counts

    def _score_one_region(self,region):
        assert(region>0 and region<=_NUM_REGIONS) #NOTE - can't score Castillo region here
        cab_counts = self._extract_region_counts(region)
    
        #rank and score the counts
        ranks={}
        for idx in range(len(cab_counts)):
            cp=cab_counts[idx]
            if cp>0:
                ranks[idx]=1
                anchor_rank=-1
                for idx2 in ranks.keys():
                    if idx!=idx2:
                        if cab_counts[idx]>cab_counts[idx2]:
                            #new one is bigger, push the old rank down
                            ranks[idx2]=ranks[idx2]+1
                        elif cab_counts[idx]==cab_counts[idx2]:
                            ranks[idx2]=ranks[idx2]+1
                            anchor_rank=ranks[idx2]
                        else:
                            ranks[idx]=ranks[idx]+1
                if anchor_rank>0:
                    ranks[idx]=anchor_rank
    
        #score now for all players
        final_scores=np.full(self._num_players,0)
        for k in ranks.keys():
            if ranks[k]>0 and ranks[k]<=3:
                final_scores[k]=self._rewards[region-1][ranks[k]-1]
            if ranks[k]==1:
                if self._grandes[k]==region:
                    final_scores[k]=final_scores[k]+2
                if self._king==region:
                    final_scores[k]=final_scores[k]+2
        #print("rewards for region "+str(region))
        #print(final_scores)
        return final_scores
    
    def _score_all_regions(self):
        final_scores=np.full(self._num_players,0)
        for r in range(_NUM_REGIONS):
            final_scores = final_scores+self._score_one_region(r+1)
        return final_scores
    
    def _all_moved(self):
        #nothing left in castillo
        for i in range(self._num_players):
            if self._board[i]!=_ZERO_CHAR:
                return False
        return True
  
    # OpenSpiel (PySpiel) API functions are below. These need to be provided by
    # every game. Some not-often-used methods have been omitted.

    def current_player(self):
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def legal_actions(self, player=None):
        """Returns a list of legal actions, sorted in ascending order.
        Args:
          player: the player whose legal moves
        Returns:
          A list of legal moves, where each move is in [0, num_distinct_actions - 1]
          at non-terminal states, and empty list at terminal states.
        """

        if player is not None and player != self._cur_player:
            return []
        elif self.is_terminal():
            return []
        else:
            return range(_ACT_END)

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
            #TODO: code to generate the actual legal actions
            mask=[0]*_ACT_END
            if self._phase_name()=='start':
                mask[_ACT_DEAL]=1
                return mask
            elif self._phase_name()=='power':
                for i in range(_NUM_POWER_CARDS):
                    mask[_ACT_POWERS+i]=1
                return mask
            elif self._phase_name()=='action':
                for i in range(_NUM_ACTION_CARDS):
                    mask[_ACT_CARDS+i]=1
                return mask
            elif self._phase_name()=='actionchoose':
                mask[_ACT_DECIDE_CAB]=1
                mask[_ACT_DECIDE_ACT]=1
            elif self._phase_name()=='actioncard':
            elif self._phase_name()=='actioncab':
            else:
                #must be score - choose a secret region
                for i in range(_NUM_REGIONS):
                    mask[_ACT_CHOOSE_SECRETS+i]=1
                return mask
                

    def apply_action(self, action):
        """Applies the specified action to the state"""

        #TODO - state changes due to specific action
    
        if self._is_game_end():
            final_scores = self._score_all_regions()
            #win points normalised between 0 and 1
            min_score = min(final_scores)
            divisor = max(final_scores)-min_score
            self._win_points = [(f-min_score)/divisor for f in final_scores]
            self._cur_player = pyspiel.PlayerId.TERMINAL
            self._is_terminal=True
        else:
            self._cur_player = self._get_next_player()
    
    def action_to_string(self, arg0, arg1=None):
        """Action -> string. Args either (player, action) or (action)."""
        player = self.current_player() if arg1 is None else arg0
        action = arg0 if arg1 is None else arg1
        #TODO - code this
        return "{} ({})".format(player,action)

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
        if self._phase_name()=='start':
            return True
        else:
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

    def chance_outcomes(self):
        return []

    def get_game(self):
        return self._game

    def get_type(self):
        return self._game.get_type()

    def serialize(self):
        return pickle.dumps(self)

    def resample_from_infostate(self):
        return [self.clone()]

    def __str__(self):
        return self._board

    def clone(self):
        #TODO - if game is changed to inherit from pyspiel, this might need changing
        return copy.deepcopy(self)
        


class ElGrandeGame(object):
    """El Grande Game
    """

    def __init__(self):
        pass

    def new_initial_state(self, players=_DEFAULT_PLAYERS):
        return ElGrandeGameState(self,players)

    def num_distinct_actions(self):
        return _ACT_END

    def policy_tensor_shape(self):
        return (_ACT_END, 1)

    def clone(self):
        return ElGrandeGame()

    def max_chance_outcomes(self):
        return _MAX_TURNS * _NUM_FULL_DECKS #chance outcomes for card dealing

    def get_parameters(self):
        return {}

    def num_players(self):
        return self._num_players

    def min_utility(self):
        return 0.0

    def max_utility(self):
        return 1.0

    def get_type(self):
        return pyspiel.GameType(
            short_name="python_el_grande_game",
            long_name="Python El Grande Game",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
            information=pyspiel.GameType.Information.PERFECT_INFORMATION, #secret choice steps implemented as sub-games
            utility=pyspiel.GameType.Utility.CONSTANT_SUM, #note: maybe this should be GENERAL_SUM?
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=_MAX_PLAYERS,
            min_num_players=1,
            provides_information_state_string=True,
            provides_information_state_tensor=True,
            provides_observation_string=True,
            provides_observation_tensor=True,
            parameter_specification={},
        )

    def utility_sum(self):
        return 0.0

    def observation_tensor_layout(self):
        return pyspiel.TensorLayout.CHW

    def deserialize_state(self, string):
        return pickle.loads(string)

    def max_game_length(self):
        #9 turns, 6 phases per turn, 5 players
        return _MAX_PLAYERS * _MAX_TURNS * _NUM_PHASES

    def __str__(self):
        return "python_el_grande_game"

    def make_py_observer(self, iig_obs_type, params):
        if params:
            raise ValueError("Params not supported")
        return ElGrandeGameObserver()


class ElGrandeGameObserver:
    """Observer, conforming to the PyObserver interface (see observer.py)."""

    def __init__(self):
        self._obs = np.zeros((_ST_IDCH, _ST_IDX_END, _ST_IDY_END), np.float32)
        self.tensor = np.ravel(self._obs)
        self.dict = {"observation": self._obs}

    def set_from(self, state, player):
        del player
        for channel in range(_ST_IDCH):
            chmat = (self._state_matrix >> channel)%2
            self._obs[channel,:,:]=chmat
        #TODO - check this

    def string_from(self, state, player):
        del player
        return str(state)
