import copy
import pickle

import numpy as np
import json

import pyspiel
import el_grande_pieces as pieces

_MAX_PLAYERS = 5
_NUM_REGIONS = pieces._NUM_REGIONS
_ZERO_CHAR = 'A'
#region order "Aragon","Castilla la Nueva","Castilla la Vieja","Cataluna","Galicia","Granada","Pais Vasco","Sevilla","Valencia"
#_DEFAULT_REGION_REWARDS = [(5,4,1),(7,4,2),(6,4,2),(4,2,1),(4,2,0),(6,3,1),(5,3,1),(4,3,1),(5,3,2)]
#_DEFAULT_PLAYERS = 4
#_DEFAULT_BOARD = 'BCDEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
#_DEFAULT_GRANDES = [1,1,1,1]
#_DEFAULT_KING = 1
_DEFAULT_STATE = '{"players":4,"rewards":{{5,4,1},{7,4,2},{6,4,2},{4,2,1},{4,2,0},{6,3,1},{5,3,1},{4,3,1},{5,3,2}},"board":"BBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","grandes":{1,1,1,1},"king":1}'


_GAME_TYPE = pyspiel.GameType(
    short_name="castillo_game",
    long_name="El Grande Castillo Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION, 
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_MAX_PLAYERS,
    min_num_players=2,
    provides_information_state_string=False,
    provides_information_state_tensor=True,
    provides_observation_string=False,
    provides_observation_tensor=False,
    provides_factored_observation_string=False)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_REGIONS,
    max_chance_outcomes=0,
    num_players=5,
    min_utility=-10.0,
    max_utility=10.0,
    utility_sum=0.0,
    max_game_length=_MAX_PLAYERS)


class CastilloGameState(pyspiel.State):
    """El Grande Castillo Game - subgame of El Grande which decides on the optimum positioning of
    pieces moved from the Castillo in the scoring phase of rounds 3/6/9
    Parameters:
    players - number of players between 1 and 5
    rewards - array of 1st/2nd/3rd place triples, one per region
    board - string of lowercase alpha characters representing numbers of caballeros per player/region: 'A'==0 and region 0 has castillo cabs
    grandes - array of grande region IDs from 1.._NUM_REGIONS
    king - region ID of king
    """

    def __init__(self, game):
        super().__init__(self,game)
        self._game = game
        self._cur_player = 0
        gameJSON=json.loads(game._parent_game_state)
        self._num_players = gameJSON["players"]
        self._is_terminal = False
        self._history = []
        self._board = gameJSON["board"]
        self._scores = gameJSON["scores"]
        self._win_points = np.full(self._num_players,0)
        self._rewards = gameJSON["rewards"]
        self._grandes = gameJSON["grandes"]
        self._king = gameJSON["king"]
 
        assert(len(self._board)==self._num_players*(len(self._rewards)))
        assert(len(self._grandes)==self._num_players)
        assert(len(self._scores)==self._num_players)
        assert(min(self._grandes)>0 and max(self._grandes)<=_NUM_REGIONS)
        assert(self._king>0 and self._king<=_NUM_REGIONS)

    # Helper functions (not part of the OpenSpiel API).

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
            return [r for r in range(_NUM_REGIONS) if (r+1)!=self._king]

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
            return [1]*_NUM_REGIONS

    def do_apply_action(self, action):
        """Applies the specified action to the state. Action+1 = regionID of region to be moved to"""

        castillo_count=ord(self._board[self._cur_player]) - ord(_ZERO_CHAR)
        region_idx=(action+1)*self._num_players+self._cur_player
        #set castillo count for this player to 0, and update region count
        self._board = self._board[:self._cur_player] + _ZERO_CHAR + self._board[(self._cur_player+1):]
        self._board = self._board[:region_idx] + chr(ord(self._board[region_idx])+castillo_count) + self._board[(region_idx+1):]
        self._history.append(action)
    
        if self._all_moved():
            final_scores = self._scores + self._score_all_regions()
            
            # turn scores into win points
            fscore=max(final_scores)
            sscore=0
            if len(final_scores)>1: #always true except in test games
                sscore = sorted(final_scores,reverse=True)[1]
            scorepoint=(fscore+sscore)/2
            #players are scored relative to the midpoint between first and second place
            self._win_points = final_scores - scorepoint

            self._cur_player = pyspiel.PlayerId.TERMINAL
            self._is_terminal=True
        else:
            self._cur_player = self._cur_player+1
    
    def action_to_string(self, arg0, arg1=None):
        """Action -> string. Args either (player, action) or (action)."""
        player = self.current_player() if arg1 is None else arg0
        action = arg0 if arg1 is None else arg1
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
        return _NUM_REGIONS

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
        n = self._num_players
        boardret = [self._board[i:i+n] for i in range(0,len(self._board),n)]
        return " ".join(boardret) + " | " + "".join([str(g) for g in self._grandes]) + " | " + str(self._king)

    def clone(self):
        #return copy.deepcopy(self)
        myclone = CastilloGameState(self._game)
        myclone._cur_player = self._cur_player
        myclone._is_terminal = self._is_terminal
        myclone._history = self._history.copy()
        myclone._board = self._board
        myclone._scores = self._scores.copy()
        myclone._win_points = self._win_points.copy()
        return myclone
    


class CastilloGame(pyspiel.Game):
    """El Grande Castillo Game - subgame of El Grande which decides on the optimum positioning of
    pieces moved from the Castillo in the scoring phase of rounds 3/6/9
    Parameters:
    players - number of players between 1 and 5
    rewards - array of 1st/2nd/3rd place triples, one per region
    board - string of lowercase alpha characters representing numbers of caballeros per player/region: 'A'==0 and region 0 has castillo cabs
    grandes - array of grande region IDs from 1.._NUM_REGIONS
    king - region ID of king
    """

    def __init__(self,params={"state":pyspiel.GameParameter('')}):
        super().__init__(self, _GAME_TYPE, _GAME_INFO, params or dict())
        #state input as json with keys players,rewards,board,grandes,king
        if params.get("state",None) is not None:
            state=params["state"].string_value() 
            if state=='':
                state=_DEFAULT_STATE
            self._parent_game_state=state 

    def new_initial_state(self):
        return CastilloGameState(self)

    def num_distinct_actions(self):
        return _NUM_REGIONS

    def policy_tensor_shape(self):
        return (_NUM_REGIONS, 1)

    def clone(self):
        return CastilloGame()

    def max_chance_outcomes(self):
        return 0

    def get_parameters(self):
        return {}

    def num_players(self):
        return self._num_players

    def min_utility(self):
        return -10.0

    def max_utility(self):
        return 10.0

    def get_type(self):
        return pyspiel.GameType(
            short_name="python_castillo_game",
            long_name="Python Castillo Game",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.CONSTANT_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=_MAX_PLAYERS,
            min_num_players=1,
            provides_information_state_string=True,
            provides_information_state_tensor=False,
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
        return self._num_players

    def __str__(self):
        return "python_castillo_game"

    def make_py_observer(self, iig_obs_type, params):
        if params:
            raise ValueError("Params not supported")
        return CastilloGameObserver()


class CastilloGameObserver:
    """Observer, conforming to the PyObserver interface (see observer.py)."""

    def __init__(self):
        self._obs = np.zeros((3, 3, 3), np.float32)
        self.tensor = np.ravel(self._obs)
        self.dict = {"observation": self._obs}

    def set_from(self, state, player):
        del player
        self._obs[:] = 0
        #TODO - check this

    def string_from(self, state, player):
        del player
        return ""
   #     return str(state)
