import copy
import pickle
import json

import numpy as np
import random
import couchdb
import pyspiel
import el_grande_pieces as pieces

_MAX_PLAYERS = 5
_NUM_REGIONS = pieces._NUM_REGIONS
_NUM_EXT_REGIONS = _NUM_REGIONS + 1
_NUM_CAB_AREAS = _NUM_EXT_REGIONS + 2 #areas caballeros can be placed, including court and province
_NUM_FULL_DECKS = pieces._NUM_FULL_DECKS
_NUM_PLAYABLE_DECKS = _NUM_FULL_DECKS + 1
_MAX_DECK_COUNT = pieces._MAX_DECK_COUNT
_NUM_ACTION_CARDS = pieces._NUM_ACTION_CARDS
_NUM_POWER_CARDS = pieces._NUM_POWER_CARDS
_POWER_CABS = pieces._POWER_CABS
_DECK_ENDS = pieces._DECK_ENDS


#game phases start,power,action,actionchoose,actioncard1,actioncab1,actioncard2,actioncab2,scoring,end
#of these, start,power,action and score can be loaded in from TTS game
_NUM_PHASES = 7
_PHASE_NAMES = ['start','power','action','actionchoose','actioncard1','actioncab1','actioncard2','actioncab2','scoring','end']
_ACTION_TYPES = ['none','move','score','power','grande','scoreboard','king','uniquescore']
_MAX_TURNS = 9 #full game has 9 turns, but we can specify fewer if need be
_SCORING_ROUND = [False,False,False,True,False,False,True,False,False,True]
_NUM_SCOREBOARDS = 2

_DEFAULT_PLAYERS = 2


#State matrix indicators
#Board state - indicators for caballero count,grande and king placement, and secret region vote

_ST_BDX_REGIONS = 0 #start of regions
_ST_BDX_CASTILLO = pieces._CASTILLO
_ST_BDX_COURT = pieces._COURT
_ST_BDX_PROVINCE = pieces._PROVINCE
_ST_BDX_END = _ST_BDX_PROVINCE + 1

_ST_BDY_CABS = 0 #start of cab counts in region columns (per-player)
_ST_BDY_GRANDE_KING = _ST_BDY_CABS + _MAX_PLAYERS # grande/king state (1 bit per player, 1 for the king)
_ST_BDY_SECRET = _ST_BDY_GRANDE_KING + 1 #secret region vote (1 bit per player)
_ST_BDY_END = _ST_BDY_SECRET + _MAX_PLAYERS

_ST_MASK_KING = _MAX_PLAYERS #in the king/grande mask, king is at end

#Action card state - int indicators per-card 
_ST_AC_UNPLAYED = 0
_ST_AC_DEALT = 1 
_ST_AC_CHOSEN = 2
_ST_AC_DONE = 3 

#Action card type - 0=instant, 1=1-step, 2=multi-step
_ST_STEPS_0 = 0
_ST_STEPS_1 = 1
_ST_STEPS_MULTI = 2

#phases in order
#_PHASE_NAMES = ['start','power','action','actionchoose','actioncard1','actioncab1','actioncard2','actioncab2','scoring','end']
_ST_PHASE_START=0
_ST_PHASE_POWER=1
_ST_PHASE_ACTION=2
_ST_PHASE_CHOOSE=3
_ST_PHASE_CARD1=4
_ST_PHASE_CAB1=5
_ST_PHASE_CARD2=6
_ST_PHASE_CAB2=7
_ST_PHASE_SCORE=8
_ST_PHASE_END=9

#Power cards are 13 bitwise indicators (1 bit per player) for power and powerpast

#Turn state (integers)
_ST_TN_ROUND = 0 
_ST_TN_PHASE = 1
_ST_TN_ACTIONTYPE = 2 #which of the possible types of card actions are being done?
_ST_TN_MOVESLEFT = 3 #for multimove phases
_ST_TN_SCORES = 4 #scores of player1,player2,...
_ST_TN_END = _ST_TN_SCORES + _MAX_PLAYERS

_ST_IDCH = 32 #highest number in the matrix should be 2^5

#list of action numbers from 0 up

_ACT_CARDS = 0 #start of 'select a card' actions
_ACT_POWERS = _ACT_CARDS + _NUM_ACTION_CARDS #select power cards
_ACT_RETRIEVE_POWERS = _ACT_POWERS + _NUM_POWER_CARDS #get back an old power card
_ACT_DECIDE_CAB = _ACT_RETRIEVE_POWERS + _NUM_POWER_CARDS #decide to place cabs first
_ACT_DECIDE_ACT = _ACT_DECIDE_CAB + 1 #decide to play the action card first
_ACT_DECIDE_ACT_ALT = _ACT_DECIDE_ACT + 1 #special for the 'OR' card - decide on the second action
_ACT_CHOOSE_SECRETS = _ACT_DECIDE_ACT_ALT + 1 #choose one secret region
_ACT_MOVE_GRANDES = _ACT_CHOOSE_SECRETS + _NUM_REGIONS
_ACT_MOVE_KINGS = _ACT_MOVE_GRANDES + _NUM_REGIONS
_ACT_MOVE_SCOREBOARDS = _ACT_MOVE_KINGS + _NUM_REGIONS
_ACT_CAB_MOVES = _ACT_MOVE_SCOREBOARDS + (_NUM_SCOREBOARDS*_NUM_REGIONS)
_ACT_SKIP = _ACT_CAB_MOVES + (_NUM_CAB_AREAS * _NUM_CAB_AREAS * _MAX_PLAYERS) #combos of moving a cab from region, to region, of player
_ACT_TRIGGER = _ACT_SKIP + 1 #explicitly trigger an instant card action
_ACT_END = _ACT_TRIGGER + 1 



_GAME_TYPE = pyspiel.GameType(
    short_name="el_grande",
    long_name="El Grande",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    #chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION, #not strictly true for deterministic chance mode, but close
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
    num_distinct_actions=_ACT_END,  
    max_chance_outcomes=np.product(_DECK_ENDS),  #Deck counts 11,9,11,11,1 
    num_players=5,
    min_utility=0.0,
    max_utility=1.0,
    utility_sum=1.0,
    max_game_length=_MAX_PLAYERS*_MAX_TURNS*_NUM_PHASES + 15*_NUM_PHASES + 40) #total phases + cabs out + card sub-actions 

class ElGrandeGameState(pyspiel.State):
    """El Grande Game in open_spiel format
    """

    def __init__(self, game):
        super().__init__(self,game)
        self._game = game
        self._cur_player = 0
        self._num_players = game._num_players
        self._game_state = game._game_state
        self._is_terminal = False
        self._history = []
        self._winner = False
        #self._dealing = True #for games with card dealing on the fly and CHANCE mode on
        self._players = []
        self._end_turn = _MAX_TURNS #default end turn is 9
        if self._game_state != '':
            self._load_game_state(self._game_state)
        else:
            #start a game with a random player assortment
            self._players=["P"+str(i) for i in range(self._num_players)]
            self._generate_board({'Players':self._players})

    # Helper functions (not part of the OpenSpiel API).
    
    #info about card names and abilities, region names, player colours
    
    def _get_rid(self,regionName):
        return pieces._REGIONS.index(regionName)
    
    def _get_pid(self,playerName):
        return self._players.index(playerName)
    
    def _get_player_name(self,pid):
        assert(pid<self._num_players)
        return self._players[pid]

    def _get_cid(self,cardName):
        return pieces._CARDS[cardName]['idx']

    def _get_phaseid(self,phaseName):
        return _PHASE_NAMES.index(phaseName)

    def _get_current_card(self):
        ccard = [c for c in range(_NUM_ACTION_CARDS) if self._acard_state[c] == _ST_AC_CHOSEN]
        assert(len(ccard)<=1) #can have at most one card chosen at a time
        if len(ccard)==0:
            return None
        else:
            return pieces._CARDS[pieces._CARDTRACK[ccard[0]]]

    def _get_round(self):
        return self._turn_state[_ST_TN_ROUND]

    def _get_current_phase_name(self):
        return _PHASE_NAMES[self._turn_state[_ST_TN_PHASE]]

    def _update_current_card_status(self,status):
        #work out which is the current action card, and set it to new status
        ccard = [c for c in range(_NUM_ACTION_CARDS) if self._acard_state[c] == _ST_AC_CHOSEN]
        assert(len(ccard)==1) #raise error if there isn't a card to update
        self._acard_state[ccard[0]] = status
       
    def _card_moves(self):
        # return 0==instant,1=1-step,2=2 or more
        card = self._get_current_card()
        if card['actiontype']=='move':
            fromreg=card['details']['from']['region']
            toreg=card['details']['to']['region']
            if fromreg=='ownerchooseplus':
                #Angry King
                return _ST_STEPS_0
            if fromreg in ['court','province'] and toreg in ['court','province']:
                #Decay, Decayall, Court
                return _ST_STEPS_0 
            if fromreg=='ownerchoose' or toreg=='ownerchoose':
                return _ST_STEPS_1
        elif card['actiontype']=='score':
            if card['details'].get('region','')=='selfchoose':
                return _ST_STEPS_1
            else:
                return _ST_STEPS_0
        elif card['actiontype']=='uniquescore':
            return _ST_STEPS_1
        elif card['actiontype'] in ['scoreboard','power','grande','king']:
            return _ST_STEPS_1

        #default is a multi-move, since this is the most common
        return _ST_STEPS_MULTI 

    def _blank_board(self,deal=False):
        self._board_state = np.full((_ST_BDX_END,_ST_BDY_END),0)
        self._acard_state = np.full(_NUM_ACTION_CARDS,0)
        self._acard_round = np.full(_NUM_ACTION_CARDS,0)
        if deal:
            self._shuffle_acards() #for deterministic 'deal-at-start' card placement
        self._pcard_state = np.full(_NUM_POWER_CARDS,0)
        self._past_pcard_state = np.full(_NUM_POWER_CARDS,0)
        self._turn_state = np.full(_ST_TN_END,0)
        self._rewards = pieces._POINTS.copy() 
        self._scoreboards = pieces._SCOREBOARDS.copy()
        self._init_move_info()

    def _shuffle_acards(self):
        for i in range(_NUM_FULL_DECKS):
            deckname='Deck'+str(i+1)
            order = pieces._DECKTRACK[deckname].copy()
            random.shuffle(order)
            for j in range(len(order)):
                cid = self._get_cid(order[j])
                self._acard_round[cid]=j+1
                if j==0:
                    self._acard_state[cid] = _ST_AC_DEALT
        #final deck
        self._acard_round[-1] = 1
        self._acard_state[-1] = _ST_AC_DEALT

    def _generate_board(self,jsonData):
        self._blank_board(True)
        self._state_add_players(jsonData["Players"])
        #random allocate players to region, put king,cabs and grandes down
        regions = [i for i in range(_NUM_REGIONS)]
        for i in range(self._num_players):
            region = random.choice(regions)
            self._board_state[region,i]=2
            self._board_state[pieces._COURT,i]=7
            self._board_state[pieces._PROVINCE,i]=21
            self._board_state[region,_ST_BDY_GRANDE_KING] |= (pow(2,i))
            regions = [i for i in regions if i!=region]
        king_region = random.choice(regions)
        self._board_state[king_region,_ST_BDY_GRANDE_KING] |= (pow(2,_ST_MASK_KING))
        self._turn_state[_ST_TN_ROUND]=1
        self._turn_state[_ST_TN_PHASE] = _ST_PHASE_POWER 

    #turn all relevant state info from DB format into game format
    def _load_game_state(self,jsonData):
        self._history = []
        self._blank_board()
        self._state_add_players(jsonData['players'])
        self._state_add_king(jsonData['king'])
        self._state_add_cabs_grandes(jsonData['pieces'])
        self._state_add_deck_info(jsonData['cards'],jsonData['pastcards'],jsonData['deckpositions'],jsonData['turninfo'])
        self._state_add_turn_info(jsonData['turninfo'])

       
    def _state_add_players(self, playerData):
        self._players = playerData
        self._num_players = len(self._players)
        self._win_points = np.full(self._num_players, 0)
        self._state_returns = np.full(self._num_players, 0)
        self._playersleft = [self._get_pid(p) for p in self._players]
        self._playersdone = []

 
    def _state_add_king(self,region_name):
        region_id = self._get_rid(region_name)
        assert(region_id < _ST_BDX_CASTILLO )
        self._board_state[(_ST_BDX_REGIONS+region_id),_ST_BDY_GRANDE_KING] |= (pow(2,_ST_MASK_KING))
        
    def _region_has_king(self,region_id):
        return self._board_state[(_ST_BDX_REGIONS+region_id),_ST_BDY_GRANDE_KING] & (pow(2,_ST_MASK_KING)) == (pow(2,_ST_MASK_KING))
    
    def _king_region(self):
        return int(np.where(self._board_state[:,_ST_BDY_GRANDE_KING]& (pow(2,_ST_MASK_KING))>0)[0][0])
    
    def _state_add_cabs_grandes(self,data):
        for player_name in data.keys():
            player_id = self._get_pid(player_name)
            for key in data[player_name].keys():
                if key=="grande":
                    region_id = self._get_rid(data[player_name][key])
                    assert(region_id < _ST_BDX_CASTILLO)
                    self._board_state[(_ST_BDX_REGIONS+region_id),_ST_BDY_GRANDE_KING] |= (pow(2,player_id))
                else:
                    region_id = self._get_rid(key)
                    assert(region_id < _ST_BDX_END)
                    self._board_state[(_ST_BDX_REGIONS+region_id),_ST_BDY_CABS + player_id]=data[player_name][key]                                                                                                     

    def _region_has_grande(self,region_id,player_id):
        return self._board_state[(_ST_BDX_REGIONS+region_id),_ST_BDY_GRANDE_KING] & (pow(2,player_id)) == (pow(2,player_id))

    def _grande_region(self,player_id):
        return int(np.where(self._board_state[:,_ST_BDY_GRANDE_KING]& (pow(2,player_id))>0)[0][0])
    
    def _region_cabcount(self,region_id,player_id):
        return self._board_state[(_ST_BDX_REGIONS+region_id),_ST_BDY_CABS + player_id]
    
    def _region_is_secret_choice(self,region_id,player_id=-1):
        if player_id<0:
            player_id=self._cur_player
        return (self._board_state[region_id,_ST_BDY_SECRET] & pow(2,player_id) > 0)

    def _has_secret_region(self,player_id=-1):
        if player_id<0:
            player_id=self._cur_player
        if len(np.where(self._board_state[:,_ST_BDY_SECRET]& pow(2,player_id) > 0)[0])>0:
            return True
        else:
            return False

    def _secret_region(self,player_id=-1):
        if player_id<0:
            player_id=self._cur_player
        return np.where(self._board_state[:,_ST_BDY_SECRET]& pow(2,player_id) > 0)[0][0]

    def _set_rewards(self,new_scores):
        #support both TERMINAL and REWARD mode
        self._turn_state[_ST_TN_SCORES:_ST_TN_SCORES+self._num_players]+=new_scores
        self._state_returns = new_scores - np.mean(new_scores)

    def _current_score(self):
        return self._turn_state[_ST_TN_SCORES:_ST_TN_SCORES+self._num_players]

    def _next_score_step_player(self):
        #find another player who needs to make a secret region choice in scoring round
        #return -1 if no such player
        waiting_players = [i for i in range(self._num_players) if not self._has_secret_region(i) and self._board_state[pieces._CASTILLO,i]>0]
        if len(waiting_players)>0:
            return waiting_players[0]
        else:
            return -1

    def _state_add_deck_info(self,cards,pastcards,deckpositions,data):
        #action cards, sorted by deck
        #self._dealing=False
        round=data['round']
        for deck in deckpositions:
            for pos in range(len(deckpositions[deck])):
                cid=self._get_cid(deckpositions[deck][pos])
                self._acard_round[cid]=pos+1
                if (pos+1)==round:
                    self._acard_state[cid]=_ST_AC_DEALT
        #always indicate deck5 card as 'dealt'
        self._acard_state[-1]=_ST_AC_DEALT 
        if len(cards)>0:
            for deck in cards.keys():
                card_id = self._get_cid(cards[deck])
                #Differentiate between init, actdone and cabdone from player info
                #NOTE - currently only getting info before card chosen in act phase 
                actdone=False
                cabsdone=False
                if len(data['playersleft'])>0:
                    if len(data['actionsdone'])>0:
                        actdone=data['actionsdone'].get(data['playersleft'][0],False)
                    if len(data['cabsdone'])>0:
                        cabsdone=data['cabsdone'].get(data['playersleft'][0],False)
                if actdone and cabsdone:
                    self._acard_state[card_id]=_ST_AC_DONE
                elif not actdone and not cabsdone:
                    self._acard_state[card_id]=_ST_AC_DEALT
                else:
                    self._acard_state[card_id]=_ST_AC_CHOSEN
        #else:
            #set flag to ensure dealing cards will be the first thing we do
            #if data['phase'] in ['start','power']:
            #    self._dealing=True


        #'done' cards dealt in previous rounds
        for deck in pastcards.keys():
            for card in pastcards[deck]:
                card_id = self._get_cid(card)
                self._acard_state[card_id]=_ST_AC_DONE

        #'done' cards dealt in this round which were played by done players
        if data['phase']=='action':
            for player in data['playersdone']:
                card_id = self._get_cid(data['actioncards'][player])
                self._acard_state[card_id]=_ST_AC_DONE

    def _state_add_turn_info(self,data):
        #power cards
        if len(data['powercards'])>0:
            for player_name in data['powercards'].keys():
                player_id = self._get_pid(player_name)
                power_id = int(data['powercards'][player_name]) 
                assert((power_id-1) <= _NUM_POWER_CARDS and power_id > 0) #power_id from 1 to _NUM_POWER_CARDS
                self._pcard_state[(power_id-1)] |= pow(2,player_id)

        #past power cards
        if len(data['powerplayed'])>0:
            for player_name in data['powerplayed'].keys():
                player_id = self._get_pid(player_name)
                for power_id in data['powerplayed'][player_name]:
                    assert((power_id-1) <= _NUM_POWER_CARDS and power_id > 0) #power_id from 1 to _NUM_POWER_CARDS
                    self._past_pcard_state[(power_id-1)] |= pow(2,player_id)

        #round
        self._turn_state[_ST_TN_ROUND]=data['round']
        #phase
        self._turn_state[_ST_TN_PHASE]=self._get_phaseid(data['phase'])
        #for 'deal all at game start' implementation, don't need explicit round start phase
        if self._turn_state[_ST_TN_PHASE] == _ST_PHASE_START:
            self._turn_state[_ST_TN_PHASE] = _ST_PHASE_POWER
            #ensure Deck5 always turned over at start
            self._acard_state[-1] = _ST_AC_DEALT

        #this turn's cards already played by players are not currently being stored - card of active player is in _acard_state

        #score for each player 
        for player_name in data['scores'].keys():
            player_id = self._get_pid(player_name)
            score = data['scores'][player_name]
            self._turn_state[_ST_TN_SCORES+player_id]=score
        #player order - stored internally for ease of use, not strictly needed in _state_ elements
        if len(data['playersleft']) == 0:
            #nothing more to do in this phase, update to next
            if self._turn_state[_ST_TN_PHASE] == _ST_PHASE_POWER:
                self._update_players_after_power()
            else:
                self._update_players_after_action()
        elif self._turn_state[_ST_TN_PHASE] == _ST_PHASE_SCORE and not _SCORING_ROUND[self._turn_state[_ST_TN_ROUND]]:
            #scoring phase with no score action - skip to next
            self._turn_state[_ST_TN_PHASE] = _ST_PHASE_POWER
            self._turn_state[_ST_TN_ROUND] += 1
            self._acard_state = np.array([1 if self._acard_round[i]==self._turn_state[_ST_TN_ROUND] else self._acard_state[i] for i in range(_NUM_ACTION_CARDS)])
            self._acard_state[-1]=1
            self._playersleft=[self._get_pid(p) for p in data['playersleft']]
            self._cur_player = self._playersleft[0]
        else:
            self._playersleft=[self._get_pid(p) for p in data['playersleft']]
            self._cur_player = self._playersleft[0] if len(self._playersleft)>0 else 0
            

    #functions for doing actions
    
    def _deal_actions(self):
        #_DECK_ENDS = [11,9,11,11,1] #hard coded number of cards per deck
        dlists = []
        for deck in pieces._DECKTRACK:
            deck_id=int(deck[4]) #5th character of 'Deckn'
            cards = [i for i in range(_DECK_ENDS[deck_id-1])  if self._acard_state[self._get_cid(pieces._DECKTRACK[deck][i])]==_ST_AC_UNPLAYED]
            dlists = dlists+[cards]
        action_list = [a + _DECK_ENDS[0]*(b + _DECK_ENDS[1]*(c + _DECK_ENDS[2]*d)) for a in dlists[0] for b in dlists[1] for c in dlists[2] for d in dlists[3]]
        return sorted(action_list)

    def _deck_pos_for_action(self,action):
        deck1 = action%_DECK_ENDS[0]
        deck2 = (action%(_DECK_ENDS[0] * _DECK_ENDS[1]))//_DECK_ENDS[0]
        deck3 = (action%(_DECK_ENDS[0] * _DECK_ENDS[1] * _DECK_ENDS[2]))//(_DECK_ENDS[0] * _DECK_ENDS[1])
        deck4 = action //(_DECK_ENDS[0] * _DECK_ENDS[1] * _DECK_ENDS[2])
        deck5 = 0 # single deck5 card is always chosen
        return (deck1,deck2,deck3,deck4,deck5)
    
    def _deal_cards_from_action(self,action):
        #mark previous cards as played
        self._acard_state = [_ST_AC_UNPLAYED if c==_ST_AC_UNPLAYED else _ST_AC_DONE for c in self._acard_state]
        deck_positions = self._deck_pos_for_action(action)
        for i in range(len(deck_positions)):
            deck="Deck"+str(i+1)
            self._acard_state[self._get_cid(pieces._DECKTRACK[deck][deck_positions[i]])]=_ST_AC_DEALT
        self._turn_state[_ST_TN_PHASE]=self._get_phaseid('power')
        self._dealing=False
             
    def _cards_for_action(self,action):
        deck_positions = self._deck_pos_for_action(action)
        guids = [pieces._DECKTRACK["Deck"+str(i+1)][deck_positions[i]] for i in range(len(deck_positions))]
        return guids
             
    def _assign_power(self,power_id):
        #power_id is array position 0..12
        self._pcard_state[power_id] |= pow(2,self._cur_player)
        self._past_pcard_state[power_id] |= pow(2,self._cur_player)
        
    def _retrieve_power(self,power_id):
        self._past_pcard_state[power_id] -= pow(2,self._cur_player)
       
    def _available_powers(self):
        #power cards available to current player in array position 0..12
        #check everyone's current cards, and current player's past 
        in_deck = [p for p in range(_NUM_POWER_CARDS) if self._pcard_state[p]==0]
        unused = [p for p in range(_NUM_POWER_CARDS) if self._past_pcard_state[p] & pow(2,self._cur_player)==0]
        return [p for p in range(_NUM_POWER_CARDS) if p in in_deck and p in unused]

    def _set_secret_region(self,region_id,player_id=-1):
        if player_id==-1:
            player_id = self._cur_player
        #ensure only one set at a time
        mask = pow(2,(_MAX_PLAYERS+1))-pow(2,player_id)-1 #masks out everything except this player
        self._board_state[:_ST_BDX_COURT,_ST_BDY_SECRET] &= np.full(_ST_BDX_COURT,mask)
        self._board_state[region_id,_ST_BDY_SECRET] |= pow(2,player_id)

    def _move_grande(self,region_id):
        #mask out current player's grande from everywhere else
        #note the mask for this has to fit one entry per grande for all players, plus also king
        mask = pow(2,(_MAX_PLAYERS+2))-pow(2,self._cur_player)-1 #masks out everything except this player
        self._board_state[:_ST_BDX_COURT,_ST_BDY_GRANDE_KING] &= np.full(_ST_BDX_COURT,mask)
        self._board_state[region_id,_ST_BDY_GRANDE_KING] |= pow(2,self._cur_player)
        
    def _move_king(self,region_id):
        mask = pow(2,(_MAX_PLAYERS+2))-pow(2,_ST_MASK_KING)-1 #masks out everything except king
        self._board_state[:_ST_BDX_COURT,_ST_BDY_GRANDE_KING] &= np.full(_ST_BDX_COURT,mask)
        self._board_state[region_id,_ST_BDY_GRANDE_KING] |= pow(2,_ST_MASK_KING)

    def _move_scoreboard(self,board_id,region_id):
        points = self._scoreboards[board_id]['points']
        self._rewards[region_id]=points
        oldregion = self._scoreboards[board_id].get('region',-1)
        if oldregion>0 and oldregion<len(self._rewards):
            self._rewards[oldregion] = pieces._POINTS[oldregion]
        self._scoreboards[board_id]['region']=region_id

    def _move_one_cab(self,from_region, to_region, of_player):
        assert(self._board_state[from_region,of_player]>0)
        self._board_state[from_region,of_player] = self._board_state[from_region,of_player] - 1
        self._board_state[to_region,of_player] = self._board_state[to_region,of_player] + 1
        #ensure that a matching pattern keeps track of what moves are now allowable
        self._register_cab_moved(from_region,to_region,of_player)
      
        #ensure locks are enforced
        if self._movement_tracking['lockfrom']:
            self._movement_tracking['from']=[from_region]
        if self._movement_tracking['lockto']:
            self._movement_tracking['to']=[to_region]
 
        #check if the patterns are all filled, and if so, set moving to false
        self._check_move_patterns_filled()
         
    def _region_str(self,region_id):
        retstr = "|".join([str(i) for i in self._board_state[region_id,_ST_BDY_CABS:(_ST_BDY_CABS + self._num_players)]])
        grandes = [str(i) for i in range(self._num_players) if self._region_has_grande(region_id,i)]
        if len(grandes)>0:
            retstr = retstr + " G(" + "|".join(grandes) + ")"
        if self._region_has_king(region_id):
            retstr = retstr + " K"
        retstr = retstr.ljust(18," ")+ "-  " + pieces._REGIONS[region_id].rjust(18," ")
        if region_id < _NUM_EXT_REGIONS:
            retstr = retstr + str(self._rewards[region_id])
        return retstr
    
    def _power_str(self,player_id):
        current = np.where(self._pcard_state & pow(2,player_id) == pow(2,player_id))[0] 
        if len(current)==0:
            cstr = "X"
        else:
            cstr = str(current[0] + 1)
        past = [str(p+1) for p in np.where(self._past_pcard_state & pow(2,player_id) == pow(2,player_id))[0]]
        return cstr + " (" + ",".join(past) + ")" + "  -  playerid " + str(player_id)

    def _acard_str(self,cardpos):
        cardstr = pieces._CARDS[pieces._CARDTRACK[cardpos]]['name']
        if self._acard_state[cardpos]==_ST_AC_DEALT:
            return cardstr
        else:
            return '('+cardstr+')'
 
    def _action_str(self):
        activecards = [c for c in range(_NUM_ACTION_CARDS) if self._acard_state[c]!=_ST_AC_UNPLAYED and self._acard_state[c]!= _ST_AC_DONE]
        assert(len(activecards)<=_NUM_PLAYABLE_DECKS)
        names = [self._acard_str(c) for c in activecards]
        return "|".join(names)
    
    def _turn_info_str(self):
        return "Round "+ str(self._turn_state[_ST_TN_ROUND]) + " " + _PHASE_NAMES[self._turn_state[_ST_TN_PHASE]] + "(player "+str(self._cur_player)+") - Cumulative Scores " +str(self._current_score())    
   
    def _move_castillo_pieces(self):
        for i in range(self._num_players):
            if self._has_secret_region(i):
                region = self._secret_region(i)
                ncabs=self._board_state[pieces._CASTILLO,i]
                self._board_state[pieces._CASTILLO,i]=0
                self._board_state[region,i]+=ncabs
    
    def _unique_score(self,action):
        region = action-_ACT_CHOOSE_SECRETS
        #choose a region for other players. For now, simulating other players
        #by choosing a "reasonable" action for each other player - the one in 
        #which they score the most

        bestreg=[[] for i in range(self._num_players)]
        hiscores = np.full(self._num_players,0)
        for i in range(_NUM_REGIONS):
            scores=self._score_one_region(i)
            #throw out old 'bests' if the hiscore improved
            bestreg = [bestreg[n] if scores[n]<=hiscores[n] else [] for n in range(self._num_players)]
            hiscores = [max(hiscores[n],scores[n]) for n in range(self._num_players)]
            bestreg = [bestreg[n]+[i] if scores[n]==hiscores[n] else bestreg[n] for n in range(self._num_players)]

        regions = np.array([random.choice(bestreg[n]) if len(bestreg[n])>0 else random.choice(_NUM_REGIONS) for n in range(self._num_players)])
        regions[self._cur_player]=region

        #determine any doubles before scoring
        score_regions=[]
        for i in range(self._num_players):
            if not any(regions[np.arange(len(regions))!=i]==regions[i]):
                score_regions=score_regions+regions[i]
        #print("Scored regions:")
        #print(score_regions)
        final_scores=np.full(self._num_players,0)
        for r in score_regions:
            final_scores += self._score_one_region(r)
        self._set_rewards(final_scores)

 
    def _special_score(self,details):
        final_scores=np.full(self._num_players,0)
        choice_step=False
        region=details.get('region','')
        if region=='fours':
            for i in range(_NUM_EXT_REGIONS): 
                if self._rewards[i][0]==4:
                    final_scores = final_scores + self._score_one_region(i)
        elif region=='fives':
            for i in range(_NUM_EXT_REGIONS):
                if self._rewards[i][0]==5:
                    final_scores = final_scores + self._score_one_region(i)
        elif region=='sixsevens':
            for i in range(_NUM_EXT_REGIONS):
                if self._rewards[i][0]==6 or self._rewards[i][0]==7:
                    final_scores = final_scores + self._score_one_region(i)
        elif region=='castillo':
            final_scores = final_scores + self._score_one_region(pieces._CASTILLO)
        elif region=='locab':
            loregions=[]
            currentlow=100
            for i in range(_NUM_EXT_REGIONS):
                this_count=sum(self._board_state[i,:self._num_players])
                if this_count<currentlow and this_count>0:
                    currentlow=this_count
                    loregions=[i]
                elif this_count==currentlow:
                    loregions = loregions + [i]
            for i in loregions:
                final_scores = final_scores + self._score_one_region(i)
        elif region=='hicab':
            hiregions=[]
            currenthigh=0
            for i in range(_NUM_EXT_REGIONS):
                this_count=sum(self._board_state[i,:self._num_players])
                if this_count>currenthigh:
                    currenthigh=this_count
                    hiregions=[i]
                elif this_count==currenthigh:
                    hiregions = hiregions + [i]
            for i in hiregions:
                final_scores = final_scores + self._score_one_region(i)
        elif region=='selfchoose':
            #choose a region to score - see if we did this already
            regions = [i for i in range(_NUM_REGIONS) if self._region_is_secret_choice(i)]
            assert(len(regions)<=1)
            if len(regions)==0:
                choice_step=True
            else:
                final_scores = final_scores + self._score_one_region(regions[0])
        else:
            #last possibility is the one where we just score the top in all regions
            for i in range(_NUM_EXT_REGIONS):
                final_scores = final_scores + self._score_one_region(i,True)
                
        if not choice_step:
            self._set_rewards(final_scores)

    def _rank_region(self, region):
        assert(region>=0 and region<_NUM_EXT_REGIONS) 
        cab_counts = self._board_state[_ST_BDX_REGIONS+region,_ST_BDY_CABS : (_ST_BDY_CABS+self._num_players) ]
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
        return ranks 

    def _score_one_region(self,region,top_only=False):
        assert(region>=0 and region<_NUM_EXT_REGIONS) 
        cab_counts = self._board_state[_ST_BDX_REGIONS+region,_ST_BDY_CABS : (_ST_BDY_CABS+self._num_players) ]
    
        #rank and score the counts
        ranks = self._rank_region(region)
        #score for all players
        final_scores=np.full(self._num_players,0)
        for k in ranks.keys():
            if ranks[k]>0 and ranks[k]<=3:
                if ranks[k]==1 or (not top_only):
                    final_scores[k]=self._rewards[region][ranks[k]-1]
            if ranks[k]==1:
                if self._region_has_grande(region,k):
                    final_scores[k]=final_scores[k]+2
                if self._region_has_king(region):
                    final_scores[k]=final_scores[k]+2
        #print("rewards for region "+str(region))
        #print(final_scores)
        return final_scores
    
    def _score_all_regions(self):
        final_scores=np.full(self._num_players,0)
        for r in range(_NUM_REGIONS):
            final_scores = final_scores+self._score_one_region(r+1)
        return final_scores
    
    def _pack_court(self):
        #move the correct number of caballeros from province to court, at the point where player action commences
        power_id = np.where(self._pcard_state & pow(2,self._cur_player) == pow(2,self._cur_player))[0][0]
        n_cabs = min(_POWER_CABS[power_id],self._board_state[_ST_BDX_PROVINCE,self._cur_player])
        self._board_state[_ST_BDX_PROVINCE,self._cur_player] -= n_cabs
        self._board_state[_ST_BDX_COURT,self._cur_player] += n_cabs
                           
        
    def _setup_caballero_placement(self):
        #set correct state information for where we will be allowed to place caballeros
        self._movement_tracking['from']=[_ST_BDX_COURT]
        self._movement_tracking['lockfrom']=False
        self._movement_tracking['to']=pieces._NEIGHBORS[self._king_region()]+[_ST_BDX_CASTILLO]
        self._movement_tracking['lockto']=False
        self._movement_tracking['moving'] = True
        card = self._get_current_card()
        n_cabs = min(int(card["name"][4]),self._board_state[_ST_BDX_COURT,self._cur_player])
        pattern = {'player':self._cur_player,'allowed':True,'max':n_cabs,'min':0}
        self._movement_tracking['patterns']=[pattern]
        self._movement_tracking['prev']=[] #no movements done in this set of substeps yet
 
    def _init_move_info(self):
        self._movement_tracking = {'from':[],'to':[],'cabs':[],'patterns':[],'queue':[],'player':0,'lockfrom':False,'lockto':False,'moving':False,'prev':[],'fromcondition':0}
        
    def _setup_action(self,alt_action=0):
        #set up info to enable multi-step actions, or flag instant actions
        
        self._init_move_info() #wipe out previous info on where caballeros were/were not allowed to move
        card = self._get_current_card()
        action_type = card['actiontype']
        card_details = card.get('details',[])
        if action_type=='choose':
            action_type = card_details[alt_action]['type']
            card_details = card_details[alt_action]['details']
               
        #additional setup for cab movement action types
        if action_type == 'all':
            #only one card is an 'and' and it happens to have two move elements
            #HACK - forgot to put 'numopt' value in cards in TTS - till I fix this, shove it in here
            #TODO - fix this properly in the other code
            card_details[0]['details']['numopt']='lteq'
            card_details[1]['details']['numopt']='lteq'

            self._do_caballero_move_info(card_details[0]['details'])
            self._add_caballero_move_info(card_details[1]['details'])
            multi_step=True
        elif action_type == 'move':
            self._do_caballero_move_info(card_details)

    def _apply_secret_choice(self,action):
        #figure out what action required a secret choice, and complete it
        card = self._get_current_card()
        action_type = card['actiontype']
        if action_type == 'score':
            self._special_score(card['details'])
        elif action_type == 'uniquescore':
            self._unique_score(action)
        elif action_type == 'move':
            #Province/Provinceall/Eviction
            self._assess_secret_choices()


    def _do_caballero_move_info(self,card_details):  
        #make movable caballeros interactable
        #'from' values are court, or region of your choice
        #self._movement_tracking = {['from']={},['to']={},['patterns']={},
        #                     ['player']=None,['lockfrom']=false,['lockto']=false,['moving']=false}

        self._movement_tracking['player']=self._cur_player
        for v in ['from','to']:
            if card_details[v]['region'] in ['court','province']:
                self._movement_tracking[v]=[self._get_rid(card_details[v]['region'])]
            elif card_details[v]['region']=='selfchoose':
                #current King's region shouldn't be in the list
                the_regions=[i for i in range(_NUM_REGIONS) if not self._region_has_king(i)]
                if v=='to':
                    the_regions = the_regions + [_ST_BDX_CASTILLO]
                self._movement_tracking[v]=the_regions
            else:
                #placeholder for some sort of owner choice
                self._movement_tracking[v]=[card_details[v]['region']]

            if card_details[v]['splitopt']=="all":
                self._movement_tracking['lock'+v]=True 
            else:
                self._movement_tracking['lock'+v]=False 

        #set the pattern for cabs to move
        pattern={}
        if card_details['player']=='foreign':
            pattern['player']=self._cur_player
            pattern['allowed']=False
        if card_details['player']=='self':
            pattern['player']=self._cur_player 
            pattern['allowed']=True
        if card_details['number']<0:
            pattern['max']=150
        else:
            pattern['max']=card_details['number']
        if card_details.get('numopt','lteq')=='lteq':
            pattern['min']=0 
        else:
            pattern['min']=pattern['max']

        #assume that, yes, we are moving
        self._movement_tracking['moving']=True
        
        #there may be a condition on choice of 'from' region - check it
        condition= card_details['from'].get('condition',None)
        if condition != None and condition!='clock':
            self._movement_tracking['fromcondition']=int(card_details['from']['condition'])
        self._movement_tracking['patterns']=[pattern] 

        #for 'Provinceone' card need to do some extra processing - one of these for each player, not one and done
        if card_details['from']['region']=='selfchoose' and card_details['to']['region']=='province':
            patterns=[]
            foreign_players = [i for i in range(self._num_players) if i!=self._cur_player]
            for p in foreign_players:
                newpat=self._movement_tracking['patterns'][0].copy()
                newpat['player']=p
                newpat['allowed']=True
                patterns=patterns+[newpat]
            self._movement_tracking['patterns']=patterns  
            self._movement_tracking['moving']=True
    
       
    def _instant_move(self):
        #do a move that can be completed in one step, then null-out movement info
        fromreg=self._movement_tracking['from']
        toreg = self._movement_tracking['to']
        card = self._get_current_card()
        card_details = card['details']
        if card_details['from']['region'] =='ownerchooseplus':
            #'Angry King' card
            #work out opponent choices, ultimately by running simulations, currently by random choice.
            #no secret choice needed from current player
            players_to_choose = [(i+self._cur_player)%self._num_players for i in range(1,self._num_players)]
            number_to_send = int(card_details['number'])
            for p in players_to_choose:
                for i in range(number_to_send):
                    #send from court if possible, then wherever you have most cabs
                    if self._board_state[pieces._COURT,p]>0:
                        self._move_one_cab(pieces._COURT, pieces._PROVINCE,p)
                    else:
                        best_regions=np.where(self._board_state[:_NUM_EXT_REGIONS,p]==max(self._board_state[:_NUM_EXT_REGIONS,p]))        
                        if max(self._board_state[:_NUM_EXT_REGIONS,p])>0:
                            self._move_one_cab(random.choice(best_regions[0]), pieces._PROVINCE,p)

        else:
            #'Decay','Decayall' and 'Court' 
            sendCount=card_details['number']
            for pl in range(self._num_players):
                sendThis=True
                if (card_details['player']=='foreign' and pl==self._cur_player) or  (card_details['player']=='self' and pl!=self._cur_player):
                    sendThis=False
                if sendThis:
                    if sendCount<0:
                        #figure out how much the 'all' in 'send all' is
                        ncabs= self._board_state[fromreg[0],_ST_BDY_CABS+pl]
                    else:
                        ncabs = min(sendCount,self._board_state[fromreg[0],_ST_BDY_CABS+pl]) 
                    self._board_state[fromreg[0],_ST_BDY_CABS+pl] -= ncabs
                    self._board_state[toreg[0],_ST_BDY_CABS+pl] += ncabs

            #null-out movement info, since we've done the move
            self._init_move_info()

 
    def _add_caballero_move_info(self,card_details):         
        pattern = {}
        if card_details['player']=='foreign':
            pattern['player']=self._cur_player
            pattern['allowed']=False
        elif card_details['player']=='self':
            pattern['player']=self._cur_player
            pattern['allowed']=True
            
        if card_details['number']<0:
            pattern['max']=150 
        else:
            pattern['max']=card_details['number']
        if card_details['numopt']=='lteq':
            pattern['min']=0 
        else:
            pattern['min']=pattern['max']        
        self._movement_tracking['patterns'] = self._movement_tracking.get('patterns',[]) + [pattern]



    def _assess_secret_choices(self):
        #work out opponent choices, ultimately by running simulations, currently by random choice.
        #ask for secret choices if you haven't already done so 
        regions = [i for i in range(_NUM_REGIONS) if self._region_is_secret_choice(i)]
        assert(len(regions)<=1)
        card=self._get_current_card()
        #code for 'Eviction' card
        if card['actiontype']=='move' and card['details']['to']['region']=='ownerchoose':
            #for each opponent, choose a region for simple maximising of their returns
            region_points = [self._score_one_region(i) for i in range(_NUM_REGIONS)]
            for i in range(self._num_players):
                if i!=self._cur_player and self._board_state[regions[0],i]>0:
                    #figure out the most lucrative place to put cabs
                    ncabs=self._board_state[regions[0],i]
                    ts = self.clone()
                    ts._board_state[:_NUM_REGIONS,i]+=ncabs
                    ts_points = [ts._score_one_region(i) for i in range(_NUM_REGIONS)]
                    improvement=np.array(ts_points)-np.array(region_points)
                    imp_i=improvement[:,i]
                    imp_i[self._king_region()]=-100 #ensure king region is never 'best'
                    best_reg = np.where(imp_i==max(imp_i))[0]
                    chosen = random.choice(best_reg)
                    self._board_state[regions[0],i] = 0
                    self._board_state[chosen,i] += ncabs
        elif card['actiontype']=='move' and card['details']['to']['region']=='province':
            #all players sending cabs to province  
            #for each opponent, choose a region for simple minimising of their losses 
            #TODO - edge cases
            region_points = [self._score_one_region(i) for i in range(_NUM_REGIONS)]
            chosen_regions = np.full(self._num_players,-1)
            for i in range(self._num_players):
                if i==self._cur_player:
                    chosen=regions[0]
                else:
                    ts = self.clone()
                    if self._movement_tracking.get('fromcondition',0)==2:
                        allowed = [ts._board_state[j][i]>=2 for j in range(_NUM_REGIONS)]
                        ts._board_state[:_NUM_REGIONS,i]-=2
                    else:
                        allowed = [ts._board_state[j][i]>=1 for j in range(_NUM_REGIONS)]
                        ts._board_state[:_NUM_REGIONS,i]=0
                    allowed[self._king_region()]=False
                    ts_points = [ts._score_one_region(i) for i in range(_NUM_REGIONS)]
                    loss=np.array(region_points) - np.array(ts_points)
                    #set loss to effective infinity in disallowed regions
                    loss_i=loss[:,i]
                    loss_i=np.array([loss_i[j] if allowed[j] else 1000 for j in range(_NUM_REGIONS)])
                    best_reg = np.where(loss_i==min(loss_i))[0]  
                    chosen = random.choice(best_reg)
                if self._movement_tracking['fromcondition']==2:
                    self._board_state[pieces._PROVINCE,i]+=2
                    self._board_state[chosen,i] -= 2
                else:
                    ncabs = self._board_state[chosen,i]
                    self._board_state[pieces._PROVINCE,i]+=ncabs
                    self._board_state[chosen,i] -=ncabs
 
        self._init_move_info()
        return []

    def _king_placement_details(self,card_details):
        if card_details['condition']=='all':
            return [_ACT_MOVE_KINGS+i for i in range(_NUM_REGIONS)]
        else:
            return [_ACT_MOVE_KINGS+i for i in range(_NUM_REGIONS) if i in pieces._NEIGHBORS[self._king_region()]]


    def _set_valid_cab_movements(self,fromcard=True):
        #use movement tracking info to determine which from/to moves are okay
        actions=[]
        #if secret choice is involved, interrupt to do this
        ccard=self._get_current_card()
        if fromcard and ccard['actiontype']=='move':
            checkregionfrom = ccard['details']['from']['region']
            checkregionto = ccard['details']['to']['region']
            if checkregionfrom =='ownerchoose' or checkregionto == 'ownerchoose':
                actions = [i + _ACT_CHOOSE_SECRETS for i in range(_NUM_REGIONS) if self._board_state[i,self._cur_player]>=self._movement_tracking.get('fromcondition',0)]
                return sorted(actions)
        for fromreg in self._movement_tracking['from']:
            for toreg in self._movement_tracking['to']:
                players=[]
                for pattern in self._movement_tracking['patterns']:
                    #add players from this pattern if it's not filled
                    if pattern.get('cabs',0)<pattern['max']:
                        mentioned_player = pattern.get('player',-1)
                        if mentioned_player == -1:
                            #any player is okay
                            players = range(self._num_players)
                        else:
                            #a player is mentioned - see if it's include or exclude
                            if pattern['allowed']:
                                players = [i for i in range(self._num_players) if ((i in players) or i==mentioned_player)]
                            else:
                                players = [i for i in range(self._num_players) if ((i in players) or i!=mentioned_player)]
                for player in players:
                    if self._board_state[fromreg,player] >0:
                        #there is a caballero here of the correct colour, so this move action is okay
                        actions.append(_ACT_CAB_MOVES + player + _MAX_PLAYERS*(toreg + _NUM_CAB_AREAS*fromreg))
        
        #during a movement_tracking episode (eg, moving cabs around the board)
        #restrict actions so we don't get extra possibilities from one substep
        #to the next
        prev_acts = self._movement_tracking.get('prev',[])
        if len(prev_acts)>0:
            actions=list(set(actions).intersection(prev_acts))
        self._movement_tracking['prev']=actions
        return sorted(actions)

    def _register_cab_moved(self,fromreg,toreg,ofplayer):
        for pattern in self._movement_tracking['patterns']:
            if self._matched_pattern(ofplayer,pattern):
                pattern['cabs']=pattern.get('cabs',0)+1
                return

    def _check_move_patterns_filled(self):
        for pattern in self._movement_tracking['patterns']:
            if pattern.get('cabs',0)<pattern['max']:
                return
        #if we didn't find any free patterns, set moving false
        self._movement_tracking['moving']=False
        
    def _matched_pattern(self,player,pattern):
        if pattern.get('player',-1) == -1:
            return True #found an unrestricted pattern
        if pattern['player']==player and pattern['allowed']:
            return True #found a pattern that says cabs of this colour are allowed
        if pattern['player']!=player and not pattern['allowed']:
            return True #found a pattern that says cabs not of this colour are allowed
        #didn't find a match
        return False
        
        
    def _set_valid_actions_from_card(self):
        #determine what sort of action is being done, then figure out the mask
        #_ACTION_TYPES = ['none','move','score','power','grande','scoreboard','king','uniquescore']
        
        #find the card that's ready to be played
        activecard = self._get_current_card()
        valid_action = activecard['actiontype']
        actions=[]
        if valid_action in['move','all']:
            return self._set_valid_cab_movements()
        elif valid_action=='king':
            return self._king_placement_details(activecard['details'])
        elif valid_action in ['score','uniquescore']:
            #if a 'score' action ends up here, we need to do region choice
            actions = actions + [(i + _ACT_CHOOSE_SECRETS) for i in range(_NUM_REGIONS) if not self._region_has_king(i)]
            return actions
        elif valid_action=='choose':
            #check if there is a movement pattern - if so, follow it, if not, ask whether we're on 1 or 2
            if self._movement_tracking['moving']:
                return self._set_valid_cab_movements()
            else:
                return[_ACT_DECIDE_ACT,_ACT_DECIDE_ACT_ALT]
        elif valid_action=='power':
            actions = actions + [(_ACT_RETRIEVE_POWERS + i) for i in range(_NUM_POWER_CARDS) if self._past_pcard_state[i] & pow(2,self._cur_player) > 0]
            return actions
        elif valid_action=='grande':
            actions = actions + [(i + _ACT_MOVE_GRANDES) for i in range(_NUM_REGIONS) if not self._region_has_king(i)]
            return actions
        elif valid_action=='scoreboard':
            actions = actions + [(i+(j*_NUM_REGIONS) + _ACT_MOVE_SCOREBOARDS) for i in range(_NUM_REGIONS) for j in range(_NUM_SCOREBOARDS) if not self._region_has_king(i)]
            return sorted(actions)


    def _after_power_choice(self):
        #functions to determine if we should move to the next phase and/or the next player, and who that player might be
        if len(self._playersleft) > 1:
            #set current player
            self._cur_player = self._playersleft[1]
            #move all players up one
            self._playersleft = self._playersleft[1:]
        else:
            #redo the whole queue, and move on to 'action' phase
            self._update_players_after_power()

    def _after_action_step(self):    
        #functions to determine if we should move to the next phase and/or the next player, 
        #and who that player might be
        #if still in the process of moving caballeros then don't do anything
        if self._movement_tracking.get('moving',False):
            return
        
        #if we have some cabs to move, ensure we do that
        if self._turn_state[_ST_TN_PHASE]==_ST_PHASE_CARD1:
            self._turn_state[_ST_TN_PHASE]=_ST_PHASE_CAB2
            self._setup_caballero_placement()
            return
        
        #if we were moving cabs and now have to do the action card, do that
        if self._turn_state[_ST_TN_PHASE]==_ST_PHASE_CAB1:
            self._turn_state[_ST_TN_PHASE]=_ST_PHASE_CARD2
            #set up the action except in the one case were we need to decide between act1 and act2
            if self._get_current_card()['actiontype']!='choose':
                self._setup_action() 
            return

        #otherwise we're done, update the queue however we need to
        self._update_current_card_status(_ST_AC_DONE)
        
        if len(self._playersleft) > 1:
            #set current player
            self._cur_player = self._playersleft[1]
            #move all players up one
            self._playersleft = self._playersleft[1:]
            self._turn_state[_ST_TN_PHASE]=_ST_PHASE_ACTION
        else:
            #redo the whole queue, and move to 'scoring' phase if appropriate, 
            #next power choosing phase otherwise
            if _SCORING_ROUND[self._turn_state[_ST_TN_ROUND]]:
                self._turn_state[_ST_TN_PHASE]=_ST_PHASE_SCORE
                #make sure secret region choices are all blank before we start this
                self._board_state[:,_ST_BDY_SECRET]=0
            else:
                self._update_players_after_action()
                
    def _after_score_step(self):
        #if everybody who needs to has chosen a secret region, do the scoring
        #score the castillo, move castillo pieces out, then score everything
        new_scores=self._score_one_region(pieces._CASTILLO)
        self._move_castillo_pieces()
        new_scores+=self._score_all_regions()
        self._set_rewards(new_scores)
        final_scores = self._current_score()
        self._board_state[:,_ST_BDY_SECRET]=0 
        if self._turn_state[_ST_TN_ROUND]==self._end_turn:
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
            self._update_players_after_action()

    def _update_players_after_power(self):
        powcards = {i:self._pcard_state[i] for i in range(_NUM_POWER_CARDS) if self._pcard_state[i]>0} 
        order=[]
        keys = sorted(powcards.keys(),reverse=True)
        for i in keys:
            player_id = int(np.log2(powcards[i]))
            order = order + [player_id]
        self._cur_player=order[0]
        self._playersleft=order
        self._turn_state[_ST_TN_PHASE]=self._get_phaseid('action')
        
    def _update_players_after_action(self):
        #get everything ready for next round
        powcards = {i:self._pcard_state[i] for i in range(_NUM_POWER_CARDS) if self._pcard_state[i]>0} 
        lowest = sorted(powcards.keys())[0]
        start_player = int(np.log2(powcards[lowest]))
        self._cur_player = start_player
        order = [start_player]
        for i in range(1,self._num_players):
            order = order +[(start_player+i) % self._num_players]
        self._playersleft = order
        self._turn_state[_ST_TN_PHASE]=_ST_PHASE_POWER
        self._turn_state[_ST_TN_ROUND]+=1
        self._pcard_state = np.full(_NUM_POWER_CARDS,0)
        #self._dealing = True #next state will be "chance" and we will deal cards
        #make the right cards dealt
        round = self._turn_state[_ST_TN_ROUND]
        for i in range(_NUM_ACTION_CARDS):
            if self._acard_round[i]==round:
                self._acard_state[i] = _ST_AC_DEALT
            elif self._acard_round[i]==(round-1):
                self._acard_state[i] = _ST_AC_DONE
        self._acard_state[-1] = _ST_AC_DEALT

    def scoring_order(self,player=-1):
        #order in which players will do their (secret) castillo choice - needed for castillo subgame
        if player<0:
            player=self._cur_player
        scoring_order=self._playersleft+self._playersdone
        zero_point=scoring_order.index(player)
        new_idxs={v:(scoring_order.index(v)-zero_point)%len(scoring_order) for v in scoring_order}
        return new_idxs

    def castillo_game_string(self,player=-1):
        #translate game state into CastilloGame format, for running tiny sims
        state_vals={"players":self._num_players,"rewards":self._rewards,"king":self._king_region()+1}
        #put player state into castillo game state in order, starting from current player
        #region 0 is the castillo
        board=np.full(self._num_players*_NUM_EXT_REGIONS,0)
        grandes=np.full(self._num_players,0)
        scores=np.full(self._num_players,0)
        new_idxs=self.scoring_order(player)
        for p in new_idxs:
            idx=new_idxs[p]
            for r in range(_NUM_REGIONS):
                board[(r+1)*self._num_players+idx] = self._board_state[r,p]    
            #castillo
            board[idx] = self._board_state[pieces._CASTILLO,p]
            grandes[idx] = self._grande_region(p)+1
            scores[idx] = self._current_score()[p]
        state_vals["board"]="".join([chr(ord('A')+b) for b in board])
        state_vals["grandes"]=grandes.tolist()
        state_vals["scores"]=scores.tolist()
        return json.dumps(state_vals) 

    # OpenSpiel (PySpiel) API functions are below. These need to be provided by
    # every game. Some not-often-used methods have been omitted.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        #elif self._dealing:
        #    return pyspiel.PlayerId.CHANCE
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
        if player is not None and player != self._cur_player:
            return []
        elif self.is_terminal():
            return []
        #elif self._dealing:
        #    return self._deal_actions()
        else:
            actions = []
            if self._turn_state[_ST_TN_PHASE]==_ST_PHASE_POWER:
                cards = self._available_powers()
                actions = actions + [c+_ACT_POWERS for c in cards]
            elif self._turn_state[_ST_TN_PHASE]==_ST_PHASE_ACTION:
                cards = [c for c in range(_NUM_ACTION_CARDS) if self._acard_state[c]==_ST_AC_DEALT]
                actions = actions + [c +_ACT_CARDS for c in cards]
            elif self._turn_state[_ST_TN_PHASE]==_ST_PHASE_CHOOSE:
                actions.append(_ACT_DECIDE_CAB)
                actions.append(_ACT_DECIDE_ACT)
                if self._get_current_card()['actiontype']=='choose':
                    actions.append(_ACT_DECIDE_ACT_ALT)
            elif self._turn_state[_ST_TN_PHASE] in [_ST_PHASE_CARD1,_ST_PHASE_CARD2]:
                if self._card_moves()==_ST_STEPS_0:
                    actions = [_ACT_SKIP,_ACT_TRIGGER]
                else:
                    actions = actions + self._set_valid_actions_from_card() + [_ACT_SKIP]
            elif self._turn_state[_ST_TN_PHASE] in [_ST_PHASE_CAB1,_ST_PHASE_CAB2]:
                actions = actions + self._set_valid_cab_movements(False) + [_ACT_SKIP]
            else:
                #must be score - choose a secret region (not one with the King)
                actions = actions + [(i + _ACT_CHOOSE_SECRETS) for i in range (_NUM_REGIONS) if not self._region_has_king(i)]
            
            return actions
    
    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        return []
        #if not self._dealing:
        #    raise ValueError("chance_outcomes called on a non-chance state.")
        #outcomes = self._deal_actions()
        #p = 1.0 / len(outcomes)
        #return [(o, p) for o in outcomes]

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


    def do_apply_action(self, action):
        #print("Player "+str(self._cur_player)+": "+str(action))
        """Applies the specified action to the state"""

        #possible actions: _ACT_DEAL, _ACT_CARDS (+ _NUM_ACTION_CARDS), _ACT_POWERS (+ _NUM_POWER_CARDS), _ACT_RETRIEVE_POWERS (+ _NUM_POWER_CARDS), 
        # _ACT_DECIDE_CAB, _ACT_DECIDE_ACT = _ACT_DECIDE_CAB + 1, _ACT_CHOOSE_SECRETS (+ _NUM_REGIONS), _ACT_MOVE_GRANDES (+ _NUM_REGIONS), 
        # _ACT_MOVE_KINGS (+ _NUM_REGIONS), _ACT_CAB_MOVES (+ _NUM_CAB_AREAS * _NUM_CAB_AREAS * _MAX_PLAYERS), _ACT_SKIP

        #don't apply an illegal action
        if not action in self.legal_actions():
            return

        #initialise rewards to zero
        self._set_rewards(np.full(self._num_players,0))
        #if self._dealing:
        #    self._deal_cards_from_action(action)
        #el
        if action>=_ACT_CARDS and action < _ACT_CARDS + _NUM_ACTION_CARDS:
            self._acard_state[action - _ACT_CARDS] = _ST_AC_CHOSEN
            self._turn_state[_ST_TN_PHASE] = _ST_PHASE_CHOOSE
        elif action >= _ACT_POWERS and action < _ACT_POWERS + _NUM_POWER_CARDS:
            self._assign_power(action - _ACT_POWERS)
            self._after_power_choice() #find next player to pick power card, or move on one phase
        elif action >= _ACT_RETRIEVE_POWERS and action < _ACT_RETRIEVE_POWERS + _NUM_POWER_CARDS:
            self._retrieve_power(action - _ACT_RETRIEVE_POWERS) #one-step action always
            self._after_action_step() #check if we need to move to next player, or next step, or keep playing actions
        elif action == _ACT_DECIDE_CAB:
            self._turn_state[_ST_TN_PHASE]=_ST_PHASE_CAB1
            self._pack_court()
            self._setup_caballero_placement()
        elif action in [_ACT_DECIDE_ACT,_ACT_DECIDE_ACT_ALT]:
            if self._turn_state[_ST_TN_PHASE]!=_ST_PHASE_CARD2:
                self._turn_state[_ST_TN_PHASE]=_ST_PHASE_CARD1
                self._pack_court()
            self._setup_action(1 if action==_ACT_DECIDE_ACT_ALT else 0)
        elif action >= _ACT_CHOOSE_SECRETS and action < _ACT_CHOOSE_SECRETS + _NUM_REGIONS:
            self._set_secret_region(action - _ACT_CHOOSE_SECRETS)
            if self._turn_state[_ST_TN_PHASE]==_ST_PHASE_SCORE:
                next_player = self._next_score_step_player()
                if next_player>=0:
                    self._cur_player = next_player
                else:
                    self._after_score_step()
            else:
                #if we weren't chosing for cab movement in scoring, we were choosing for a card action
                self._apply_secret_choice(action)
                self._after_action_step() 
        elif action >= _ACT_MOVE_GRANDES and action < _ACT_MOVE_GRANDES + _NUM_REGIONS:
            self._move_grande(action - _ACT_MOVE_GRANDES) #1-step action always
            self._after_action_step() 
        elif action >= _ACT_MOVE_KINGS and action < _ACT_MOVE_KINGS + _NUM_REGIONS:
            self._move_king(action - _ACT_MOVE_KINGS) #1-step action always
            self._after_action_step() 
        elif action >= _ACT_MOVE_SCOREBOARDS and action < _ACT_MOVE_SCOREBOARDS + (_NUM_SCOREBOARDS*_NUM_REGIONS):
            board = (action - _ACT_MOVE_SCOREBOARDS)//_NUM_REGIONS
            region = (action - _ACT_MOVE_SCOREBOARDS)%_NUM_REGIONS
            self._move_scoreboard(board,region)
            self._after_action_step() 
        elif action == _ACT_TRIGGER:
            #trigger an instant action
            card = self._get_current_card()
            if card['actiontype']=='score':
                self._special_score(card['details'])
            else:
                #instant actions are all either move or score
                self._instant_move()
            self._after_action_step()
        elif action == _ACT_SKIP:
            #skip an instant action
            self._init_move_info()
            self._after_action_step()
        else:
            #moving a caballero fromregion, toregion, ofplayer
            fromRegion = (action- _ACT_CAB_MOVES)//(_NUM_CAB_AREAS * _MAX_PLAYERS)
            toRegion = ((action- _ACT_CAB_MOVES)%(_NUM_CAB_AREAS * _MAX_PLAYERS))//_MAX_PLAYERS
            ofPlayer = (action- _ACT_CAB_MOVES)%_MAX_PLAYERS
            self._move_one_cab(fromRegion, toRegion, ofPlayer)        
            self._after_action_step() 
    
    
    def action_to_string(self, arg0, arg1=None, withPlayer=True):
        """Action -> string. Args either (player, action) or (action)."""
        player = self.current_player() if arg1 is None else arg0
        action = arg0 if arg1 is None else arg1
        actionString=""
        #if self._dealing:
        #    actionString = "Deal " + "|".join([pieces._CARDS[c]['name'] for c in self._cards_for_action(action)])
        #el
        if action>=_ACT_CARDS and action < _ACT_CARDS + _NUM_ACTION_CARDS:
            cardname = pieces._CARDS[pieces._CARDTRACK[action-_ACT_CARDS]]['name']
            actionString = "Action "+cardname
        elif action >= _ACT_POWERS and action < _ACT_POWERS + _NUM_POWER_CARDS:
            actionString = "Power "+str(action + 1 - _ACT_POWERS)
        elif action >= _ACT_RETRIEVE_POWERS and action < _ACT_RETRIEVE_POWERS + _NUM_POWER_CARDS:
            actionString = "Retrieve Power "+str(action + 1 - _ACT_RETRIEVE_POWERS)
        elif action == _ACT_DECIDE_CAB:
            actionString = "Decide Caballeros First"
        elif action == _ACT_DECIDE_ACT:
            actionString = "Decide Action First"
        elif action == _ACT_DECIDE_ACT_ALT:
            actionString = "Decide Alternate Action First"
        elif action >= _ACT_CHOOSE_SECRETS and action < _ACT_CHOOSE_SECRETS + _NUM_REGIONS:
            actionString = "Choose "+ pieces._REGIONS[action - _ACT_CHOOSE_SECRETS]
        elif action >= _ACT_MOVE_GRANDES and action < _ACT_MOVE_GRANDES + _NUM_REGIONS:
            actionString = "Grande to "+ pieces._REGIONS[action - _ACT_MOVE_GRANDES]
        elif action >= _ACT_MOVE_KINGS and action < _ACT_MOVE_KINGS + _NUM_REGIONS:
            actionString = "King to "+ pieces._REGIONS[action - _ACT_MOVE_KINGS]
        elif action >= _ACT_MOVE_SCOREBOARDS and action < _ACT_MOVE_SCOREBOARDS + (_NUM_SCOREBOARDS*_NUM_REGIONS):
            board = (action - _ACT_MOVE_SCOREBOARDS)//_NUM_REGIONS
            region = (action - _ACT_MOVE_SCOREBOARDS)%_NUM_REGIONS
            actionString = "Move scoreboard "+ str(self._scoreboards[board]['points']) +" to " + pieces._REGIONS[region]   
        elif action == _ACT_SKIP:
            actionString = "Skip this step"
        elif action == _ACT_TRIGGER:
            actionString = "Trigger card action"
        else:
            #moving a caballero fromregion, toregion, ofplayer
            fromRegion = (action- _ACT_CAB_MOVES)//(_NUM_CAB_AREAS * _MAX_PLAYERS)
            toRegion = ((action- _ACT_CAB_MOVES)%(_NUM_CAB_AREAS * _MAX_PLAYERS))//_MAX_PLAYERS
            ofPlayer = (action- _ACT_CAB_MOVES)%_MAX_PLAYERS
            actionString = self._players[ofPlayer] + " caballero from " + pieces._REGIONS[fromRegion] + " to " + pieces._REGIONS[toRegion]        
       
        if withPlayer: 
            return "{} ({})".format(self._players[player],actionString)
        else:
            return actionString

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        #Set to 'win_points' for TERMINAL reward model, 'state_returns' for REWARDS
        return self._win_points
        #return self._state_returns

    def rewards(self):
        return self.returns()

    def player_reward(self, player):
        return self.rewards()[player]

    def player_return(self, player):
        return self.returns()[player]

    def is_chance_node(self):
        #if self._dealing:
        #    return True
        #else:
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
        #format for state matrix - board elements with slash-separated player cab numbers, Grande player IDs, King
        #                        - power cards by player
        #                        - action cards by deck, replacing name with playerID if played
        lines = ["REGIONS"]
        for i in range(_NUM_CAB_AREAS):
            lines = lines + [self._region_str(i)]
        lines = lines + ["","POWER CARDS"]
        for i in range(self._num_players):
            lines = lines + [self._power_str(i)]
        lines = lines + ["","ACTION CARDS",self._action_str()]
        lines = lines + ["","TURN", self._turn_info_str()]
        return "\n".join(lines)


    def clone(self):
        #TODO - see if we can do this without needing explicit statements
        my_copy = self._game.new_initial_state()
        my_copy._acard_state = self._acard_state.copy()
        my_copy._acard_round = self._acard_round.copy()
        my_copy._board_state = self._board_state.copy()
        my_copy._cur_player=self._cur_player
        #my_copy._dealing = self._dealing
        my_copy._end_turn = self._end_turn
        my_copy._history=self._history.copy()
        my_copy._is_terminal=self._is_terminal
        my_copy._movement_tracking = copy.deepcopy(self._movement_tracking)
        my_copy._num_players=self._num_players
        my_copy._past_pcard_state = self._past_pcard_state.copy()
        my_copy._pcard_state = self._pcard_state.copy()
        my_copy._players = self._players.copy()
        my_copy._playersdone = self._playersdone.copy()
        my_copy._playersleft = self._playersleft.copy()
        my_copy._rewards = self._rewards.copy()
        my_copy._scoreboards = self._scoreboards.copy()
        my_copy._turn_state = self._turn_state.copy()
        my_copy._win_points = self._win_points.copy()
        my_copy._state_returns = self._state_returns.copy()
        my_copy._winner=self._winner
        return my_copy

class ElGrandeGame(pyspiel.Game):
    """El Grande Game
    """

    def __init__(self,params={"players":pyspiel.GameParameter(_DEFAULT_PLAYERS),"game_state":pyspiel.GameParameter(''),"game_state_json":pyspiel.GameParameter('')}):
        couchip = '127.0.0.1:5984'
        credentials = 'admin:elderberry'
        couch = couchdb.Server('http://'+credentials+'@'+couchip)
        super().__init__(self, _GAME_TYPE, _GAME_INFO, params or dict())
        self._num_players=4
        if params.get("players",None) is not None:
            self._num_players=params["players"].int_value()
        game_state=''
        game_state_json=''
        
        if params.get("game_state",None) is not None:
            game_state=params["game_state"].string_value()
        if params.get("game_state_json",None) is not None:
            game_state_json=params["game_state_json"].string_value()

        #there is no need for _state and _state_json to both be given as parameters - if they are, use _state_json
        if game_state == '' and game_state_json == '':
            self._game_state=''
        elif game_state_json != '':
            #parameter is actually a string - convert to json doc for compatibility
            self._game_state = json.loads(game_state_json)
        else:
            gamehistdb = couch['game_history']
            self._game_state = gamehistdb[game_state]

    def new_initial_state(self):
        return ElGrandeGameState(self)

    def num_distinct_actions(self):
        return _ACT_END

    def policy_tensor_shape(self):
        return (_ACT_END, 1)

    def clone(self):
        return ElGrandeGame()

    def max_chance_outcomes(self):
        return np.prod(_DECK_ENDS) #possible chance outcomes for card dealing

    def get_parameters(self):
        return {}

    def num_players(self):
        return self._num_players

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

    #def max_game_length(self):
    #    #9 turns, 6 phases per turn, 5 players
    #    return _MAX_PLAYERS * _MAX_TURNS * _NUM_PHASES

    def __str__(self):
        return "python_el_grande_game"

    def make_py_observer(self, iig_obs_type, params):
        if params:
            raise ValueError("Params not supported")
        return ElGrandeGameObserver()


class ElGrandeGameObserver:
    """Observer, conforming to the PyObserver interface (see observer.py).
       Board representation - 5 bits per player per region, representing 0-30 caballeros
       GK - representation of grande or king being in a region - straight from board
       Card representation - 2 bits per card - 00=Unplayed, 01=Dealt, 10=Chosen, 11=Done
       Phase representation - 000=Power/Start, 001=Action, 010=Choose, 011=Card1, 100=Cab1, 101=Card2, 110=Cab2, 111=Score/End 
    """
    _PHASES=[[0,0,0],[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],[1,1,1]]

    def __init__(self):
        board_size=(_ST_IDCH*_ST_BDX_PROVINCE*_MAX_PLAYERS)
        gk_size=(_NUM_REGIONS*(_MAX_PLAYERS+1))
        cards_size=(_NUM_ACTION_CARDS*2)
        phase_size=3

        self.tensor = np.zeros(board_size + gk_size + cards_size + phase_size, np.float32)
        self._board = self.tensor[:board_size].reshape(_ST_IDCH, _ST_BDX_PROVINCE, _MAX_PLAYERS)
        self._gk = self.tensor[board_size:(board_size+gk_size)].reshape(_NUM_REGIONS,_MAX_PLAYERS+1)
        self._cards = self.tensor[(board_size+gk_size):(board_size+gk_size+cards_size)]
        self._phase = self.tensor[(board_size+gk_size+cards_size):]
        self.dict = {"board": self._board,"grande_king":self._gk,"cards":self._cards,"phase":self._phase}



    def set_from(self, state, player):
        del player
        for channel in range(_ST_IDCH):
            chmat = (state._board_state[:_ST_BDX_PROVINCE,:_MAX_PLAYERS] >> channel)%2
            self._board[channel,:,:]=chmat

        for channel in range(_MAX_PLAYERS+1):
            chvec = (state._board_state[:_ST_BDX_CASTILLO,_ST_BDY_GRANDE_KING] >> channel)%2
            self._gk[:,channel]=chvec

        cards_ud_cd = [state._acard_state[s] in [_ST_AC_CHOSEN,_ST_AC_DONE] for s in range(_NUM_ACTION_CARDS)]
        cards_uc_dd = [state._acard_state[s] in [_ST_AC_DEALT,_ST_AC_DONE] for s in range(_NUM_ACTION_CARDS)]
        self._cards[:_NUM_ACTION_CARDS]=cards_ud_cd
        self._cards[_NUM_ACTION_CARDS:]=cards_uc_dd

        self._phase[:]=self._PHASES[state._turn_state[_ST_TN_PHASE]]

    def string_from(self, state, player):
        del player
        return str(state)

pyspiel.register_game(_GAME_TYPE,ElGrandeGame)
