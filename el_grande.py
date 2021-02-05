import copy
import pickle
import json

import numpy as np
import random

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
_POWER_CABS = [6,5,5,4,4,3,3,2,2,1,1,0,0]
#game phases start,power,action,actionchoice,actioncard,actioncabs,score
_NUM_PHASES = 7
_PHASE_NAMES = ['start','power','action','actionchoose','actioncard','actioncab','scoring','end']
_ACTION_TYPES = ['none','move','score','power','grande','scoreboard','king','uniquescore']
_MAX_TURNS = 9 #full game has 9 turns, but we can specify fewer if need be
_SCORING_ROUND = [False,False,False,True,False,False,True,False,False,True]
_NUM_SCOREBOARDS = 2
_DECK_ENDS = [11,9,11,11,1] #hard code number of cards per deck

_DEFAULT_PLAYERS = 4


#State matrix indicators
#Board state - indicators for caballero count,grande and king placement, and secret region vote

_ST_BDX_REGIONS = 0 #start of regions
_ST_BDX_CASTILLO = _ST_BDX_REGIONS + _NUM_REGIONS
_ST_BDX_COURT = _ST_BDX_CASTILLO + 1
_ST_BDX_PROVINCE = _ST_BDX_COURT + 1
_ST_BDX_END = _ST_BDX_PROVINCE + 1

_ST_BDY_CABS = 0 #start of cab counts in region columns (per-player)
_ST_BDY_GRANDE_KING = _ST_BDY_CABS + _MAX_PLAYERS # grande/king state (1 bit per player, 1 for the king)
_ST_BDY_SECRET = _ST_BDY_GRANDE_KING + 1 #secret region vote (1 bit per player)
_ST_BDY_END = _ST_BDY_SECRET + _MAX_PLAYERS

_ST_MASK_KING = _MAX_PLAYERS #in the king/grande mask, king is at end

#Action card state - int indicators per-card - unplayed, done, playable(initial,after cabsfirst,after actfirst)
_ST_AC_UNPLAYED = 0
_ST_AC_PLAY_READY = 1 #dealt out
_ST_AC_PLAY_INIT = 2 #selected by current player
_ST_AC_PLAY_CABFIRST = 3 
_ST_AC_PLAY_CABSECOND = 4 
_ST_AC_PLAY_ACTFIRST = 5 
_ST_AC_PLAY_ACTSECOND = 6 
_ST_AC_DONE = 7

#phases in order
#_PHASE_NAMES = ['start','power','action','actionchoose','actioncard','actioncab','scoring','end']
_ST_PHASE_START=0
_ST_PHASE_POWER=1
_ST_PHASE_ACTION=2
_ST_PHASE_CHOOSE=3
_ST_PHASE_CARD=4
_ST_PHASE_CAB=5
_ST_PHASE_SCORE=6
_ST_PHASE_END=7

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
_ACT_END = _ACT_SKIP + 1 

class ElGrandeGameState(object):
    """El Grande Game in open_spiel format
    """

    def __init__(self, game):
        self._game = game
        self._cur_player = 0
        self._num_players = 0
        self._is_terminal = False
        self._history = []
        self._winner = False
        self._dealing = True
        self._board_state = np.full((_ST_BDX_END,_ST_BDY_END),0)
        self._acard_state = np.full(_NUM_ACTION_CARDS,0)
        self._pcard_state = np.full(_NUM_POWER_CARDS,0)
        self._past_pcard_state = np.full(_NUM_POWER_CARDS,0)
        self._turn_state = np.full(_ST_TN_END,0)
        self._cards = {} #cards will be loaded at state initialisation
        self._regions = []
        self._players = []
        self._end_turn = _MAX_TURNS #default end turn is 9
        self._init_move_info()
        
    # Helper functions (not part of the OpenSpiel API).
    
    #info about card names and abilities, region names, player colours
    
    def _get_rid(self,regionName):
        return self._regions.index(regionName)
    
    def _get_pid(self,playerName):
        return self._players.index(playerName)

    def _get_cid(self,cardName,global_idx=True):
        if global_idx:
            #indexing 1..43
            return self._cards[cardName]['globalidx']
        else:
            return self._cards[cardName]['deckidx']

    def _get_phaseid(self,phaseName):
        return _PHASE_NAMES.index(phaseName)

    def _get_current_card(self):
        ccard = [c for c in range(_NUM_ACTION_CARDS) if self._acard_state[c] not in [_ST_AC_UNPLAYED,_ST_AC_DONE,_ST_AC_PLAY_READY]]
        assert(len(ccard)<=1) #can have at most one card in the playable states
        if len(ccard)==0:
            return None
        else:
            return self._cards[self._cardtrack[ccard[0]]]

    def _get_action_card_status(self):
        ccard = [c for c in range(_NUM_ACTION_CARDS) if self._acard_state[c] not in [_ST_AC_UNPLAYED,_ST_AC_PLAY_READY,_ST_AC_DONE]
]
        assert(len(ccard))==1
        return self._acard_state[ccard[0]]

    def _update_action_card_status(self,phase):
        #work out which is the current action card, and what its status should become
        ccard = [c for c in range(_NUM_ACTION_CARDS) if self._acard_state[c] not in [_ST_AC_UNPLAYED,_ST_AC_PLAY_READY,_ST_AC_DONE]
]
        assert(len(ccard))==1
        if phase==_ST_PHASE_CARD:
            self._acard_state[ccard[0]] = _ST_AC_PLAY_ACTFIRST if self._acard_state[ccard[0]] == _ST_AC_PLAY_INIT else _ST_AC_PLAY_ACTSECOND
        elif phase==_ST_PHASE_CAB:    
            self._acard_state[ccard[0]] = _ST_AC_PLAY_CABFIRST if self._acard_state[ccard[0]] == _ST_AC_PLAY_INIT else _ST_AC_PLAY_CABSECOND
        else:
            self._acard_state[ccard[0]] = _ST_AC_DONE
        
        self._turn_state[_ST_TN_PHASE]=phase


    def _load_game_info(self,jsonData):
        self._regiondata = jsonData["Regions"]
        self._regions = [r for r in self._regiondata.keys() if r!='Castillo']+['Castillo','court','province']
        self._init_rewards()
        self._init_neighbors()
        self._players = jsonData["Players"]
        self._num_players = len(self._players)
        self._win_points = np.full(self._num_players, 0)
        self._decktrack={}
        self._cardtrack=[]
        #keep count of global and per_deck indices as we read in the cards
        globalidx=0
        deckidx=0
        for deck in jsonData["Cards"]:
            deckidx =0
            for cardguid in jsonData["Cards"][deck]:
                globalidx += 1
                deckidx += 1
                self._cards[cardguid]=jsonData["Cards"][deck][cardguid]
                self._cards[cardguid]['globalidx']=globalidx
                self._cards[cardguid]['deckidx']=deckidx
                self._decktrack[deck]=self._decktrack.get(deck,[])+[cardguid]
                self._cardtrack=self._cardtrack+[cardguid]
        assert(len(self._cards.keys())==_NUM_ACTION_CARDS)
        #mobile scoreboard data, indexed by guid, containing points
        #translate to indexing by id, with links to guid and points
        boards=jsonData['Scoreboards']
        self._scoreboards = [{'guid':b,'points':boards[b]['points']} for b in boards]
        #start info, for game that starts from this code, not a fed-in state
        self._playersleft = [self._get_pid(p) for p in self._players]
        self._playersdone = []
        #TODO - random allocate players to region, put king,cabs and grandes down 

    #turn all relevant state info from DB format into game format
    def _load_game_state(self,jsonData):
        self._state_add_king(jsonData['king'])
        self._state_add_cabs_grandes(jsonData['pieces'])
        self._state_add_deck_info(jsonData['cards'],jsonData['pastcards'],jsonData['turninfo'])
        self._state_add_turn_info(jsonData['turninfo'])

        
    def _state_add_king(self,region_name):
        region_id = self._get_rid(region_name)
        assert(region_id < _ST_BDX_CASTILLO )
        self._board_state[(_ST_BDX_REGIONS+region_id),_ST_BDY_GRANDE_KING] |= (pow(2,_ST_MASK_KING))
        
    def _region_has_king(self,region_id):
        return self._board_state[(_ST_BDX_REGIONS+region_id),_ST_BDY_GRANDE_KING] & (pow(2,_ST_MASK_KING)) == (pow(2,_ST_MASK_KING))
        
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

    def _region_cabcount(self,region_id,player_id):
        return self._board_state[(_ST_BDX_REGIONS+region_id),_ST_BDY_CABS + player_id]

    def _state_add_deck_info(self,cards,pastcards,data):
        #action cards, sorted by deck
        if len(cards)>0:
            for deck in cards.keys():
                card_id = self._get_cid(cards[deck])
                #Differentiate between init, actdone and cabdone from player info
                #TODO - ensure this is consistent 
                actdone=False
                cabsdone=False
                if len(data['playersleft'])>0:
                    if len(data['actionsdone'])>0:
                        actdone=data['actionsdone'].get(data['playersleft'][0],False)
                    if len(data['cabsdone'])>0:
                        cabsdone=data['cabsdone'].get(data['playersleft'][0],False)
                if actdone and cabsdone:
                    self._acard_state[card_id-1]=_ST_AC_DONE
                elif actdone and not cabsdone:
                    self._acard_state[card_id-1]=_ST_AC_PLAY_CABSECOND
                    self._turn_state[_ST_TN_PHASE] = _ST_PHASE_CAB
                elif (not actdone) and cabsdone:
                    self._acard_state[card_id-1]=_ST_AC_PLAY_ACTSECOND
                    self._turn_state[_ST_TN_PHASE] = _ST_PHASE_CARD
                else:
                    self._acard_state[card_id-1]=_ST_AC_PLAY_READY
                    self._turn_state[_ST_TN_PHASE] = _ST_PHASE_CHOOSE
                    

        #'done' cards dealt in previous rounds
        for deck in pastcards.keys():
            for card in pastcards[deck]:
                card_id = self._get_cid(card)
                self._acard_state[card_id-1]=_ST_AC_DONE

        #'done' cards dealt in this round which were played by done players
        if data['phase']=='action':
            for player in data['playersdone']:
                card_id = self._get_cid(data['actioncards'][player])
                self._acard_state[card_id-1]=_ST_AC_DONE

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

        #'dealing' is always false - no info state is produced immediately before a deal
        self._dealing = False
        #this turn's cards already played by players are not currently being stored - card of active player is in _acard_state

        #score for each player 
        for player_name in data['scores'].keys():
            player_id = self._get_pid(player_name)
            score = data['scores'][player_name]
            self._turn_state[_ST_TN_SCORES+player_id]=score
        #player order - stored internally for ease of use, not strictly needed in _state_ elements
        self._playersleft=[self._get_pid(p) for p in data['playersleft']]
        self._cur_player = self._playersleft[0] if len(self._playersleft)>0 else 0
            
    def _init_neighbors(self):
        neighbors = np.full(_NUM_REGIONS,None)
        for region_id in range(_NUM_REGIONS):
            neighbors[region_id]=[self._get_rid(r) for r in self._regiondata[self._regions[region_id]]['neighbors']]
        self._neighbors = neighbors
        
    def _init_rewards(self):
        #from initial game info, not taking into account any scoreboard movement
        self._rewards = [self._regiondata[self._regions[r]]['points'][0] for r in range(_NUM_EXT_REGIONS)]
    

    #functions for doing actions
    
    def _deal_all_decks(self):
        #TODO - delete if no longer needed
        #set current cards as done
        self._acard_state = [_ST_AC_UNPLAYED if s==_ST_AC_UNPLAYED else _ST_AC_DONE for s in self._acard_state]
        #final card is always set ready after dealing (Deck5)
        self._acard_state[-1] = _ST_AC_UNPLAYED
        for deck in self._decktrack:
            card_guids = [k for k in self._decktrack[deck] if self._acard_state[self._get_cid(k)-1]==_ST_AC_UNPLAYED]
            next_card_guid = random.choice(card_guids)
            self._acard_state[self._get_cid(next_card_guid)-1]=_ST_AC_PLAY_READY
        self._turn_state[_ST_TN_PHASE]=self._get_phaseid('power')

    def _deal_actions(self):
        #_DECK_ENDS = [11,9,11,11,1] #hard coded number of cards per deck
        dlists = []
        for deck in self._decktrack:
            deck_id=int(deck[4]) #5th character of 'Deckn'
            cards = [i for i in range(_DECK_ENDS[deck_id-1])  if self._acard_state[self._get_cid(self._decktrack[deck][i])-1]==_ST_AC_UNPLAYED]
            dlists = deal_lists+[cards]
        action_list = [a + _DECK_ENDS[0]*(b + _DECK_ENDS[1]*(c + _DECK_ENDS[2]*d)) for a in dlists[0] for b in dlists[1] for c in dlists[2] for d in dlists[3]]
        return sorted(action_list)

    def _deck_pos_for_action(self,action):
        deck1 = action%_DECK_ENDS[0]
        deck2 = (action%(_DECK_ENDS[0] * _DECK_ENDS[1]))//_DECK_ENDS[0]
        deck3 = (action%(_DECK_ENDS[0] * _DECK_ENDS[1] * _DECK_ENDS[2]))//(_DECK_ENDS[0] * _DECK_ENDS[1])
        deck4 = action //(_DECK_ENDS[0] * _DECK_ENDS[1] * _DECK_ENDS[2])
        return (deck1,deck2,deck3,deck4)
    
    def _deal_cards_from_action(self,action):
        deck_positions = self.deck_pos_for_action(action)
        for i in range(len(deck_positions)):
            deck="Deck"+str(i)
            self._acard_state[self._get_cid(self._decktrack[deck][deck_positions[i]])-1]=_ST_AC_PLAY_READY
        self._turn_state[_ST_TN_PHASE]=self._get_phaseid('power')
        self._dealing=False
             
    def _cards_for_action(self,action):
        deck_positions = self.deck_pos_for_action(action)
        guids = [self._decktrack["Deck"+str(i)] for i in range(len(deck_positions))]
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
        self._board_state[:_NUM_REGIONS,_ST_BDY_SECRET] &= np.full(_NUM_REGIONS,mask)
        self._board_state[region_id,_ST_BDY_SECRET] |= pow(2,player_id)

    def _move_grande(self,region_id):
        #mask out current player's grande from everywhere else
        #note the mask for this has to fit one entry per grande for all players, plus also king
        mask = pow(2,(_MAX_PLAYERS+2))-pow(2,self._cur_player)-1 #masks out everything except this player
        self._board_state[:_NUM_REGIONS,_ST_BDY_GRANDE_KING] &= np.full(_NUM_REGIONS,mask)
        self._board_state[region_id,_ST_BDY_GRANDE_KING] |= pow(2,self._cur_player)
        
    def _move_king(self,region_id):
        mask = pow(2,(_MAX_PLAYERS+2))-pow(2,_ST_MASK_KING)-1 #masks out everything except king
        self._board_state[:_NUM_REGIONS,_ST_BDY_GRANDE_KING] &= np.full(_NUM_REGIONS,mask)
        self._board_state[region_id,_ST_BDY_GRANDE_KING] |= pow(2,_ST_MASK_KING)

    def _move_scoreboard(self,board_id,region_id):
        points = self._scoreboards[board_id]['points']
        self._rewards[region_id]=points
        self._scoreboards[board_id]['region']=region_id
        #TODO - keep consistent when scoreboard moved from another region

    def _move_one_cab(self,from_region, to_region, of_player):
        assert(self._board_state[from_region,of_player]>0)
        self._board_state[from_region,of_player] = self._board_state[from_region,of_player] - 1
        self._board_state[to_region,of_player] = self._board_state[to_region,of_player] + 1
        #ensure that a matching pattern keeps track of what moves are now allowable
        self._register_cab_moved(from_region,to_region,of_player)
       
        #check if the patterns are all filled, and if so, set moving to false
        self._check_move_patterns_filled()
         
    def _region_str(self,region_id):
        retstr = "|".join([str(i) for i in self._board_state[region_id,_ST_BDY_CABS:(_ST_BDY_CABS + self._num_players)]])
        grandes = [str(i) for i in range(self._num_players) if self._region_has_grande(region_id,i)]
        if len(grandes)>0:
            retstr = retstr + " G(" + "|".join(grandes) + ")"
        if self._region_has_king(region_id):
            retstr = retstr + " K"
        retstr = retstr + "  -  " + self._regions[region_id]
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
        cardstr = self._cards[self._cardtrack[cardpos]]['name']
        if self._acard_state[cardpos]==_ST_AC_PLAY_READY:
            return cardstr
        elif self._acard_state[cardpos]==_ST_AC_PLAY_INIT:
            return '* '+cardstr
        elif self._acard_state[cardpos]==_ST_AC_PLAY_CABFIRST:
            return 'c '+cardstr
        elif self._acard_state[cardpos]==_ST_AC_PLAY_CABSECOND:
            return 'C '+cardstr
        elif self._acard_state[cardpos]==_ST_AC_PLAY_ACTFIRST:
            return 'a '+cardstr
        elif self._acard_state[cardpos]==_ST_AC_PLAY_ACTSECOND:
            return 'A '+cardstr
        else:
            return '('+cardstr+')'
 
    def _action_str(self):
        activecards = [c for c in range(_NUM_ACTION_CARDS) if self._acard_state[c]!=_ST_AC_UNPLAYED and self._acard_state[c]!= _ST_AC_DONE]
        assert(len(activecards)<=_NUM_PLAYABLE_DECKS)
        names = [self._acard_str(c) for c in activecards]
        return "|".join(names)
    
    def _turn_info_str(self):
        return "Round "+ str(self._turn_state[_ST_TN_ROUND]) + " " + _PHASE_NAMES[self._turn_state[_ST_TN_PHASE]] + "(player "+str(self._cur_player)+")"     
    
    
    def _special_score(self,details):
        final_scores=np.full(self._num_players,0)
        choice_step=False
        if details['region']=='fours':
            for i in range(_NUM_EXT_REGIONS): 
                if self._rewards[i][0]==4:
                    final_scores = final_scores + self._score_one_region(i)
        elif details['region']=='fives':
            for i in range(_NUM_EXT_REGIONS):
                if self._rewards[i][0]==5:
                    final_scores = final_scores + self._score_one_region(i)
        elif details['region']=='sixsevens':
            for i in range(_NUM_EXT_REGIONS):
                if self._rewards[i][0]==6 or self._rewards[i][0]==7:
                    final_scores = final_scores + self._score_one_region(i)
        elif details['region']=='castillo':
                final_scores = final_scores + self._score_one_region(i)
        elif details['region']=='locab':
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
        elif details['region']=='hicab':
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
        elif details['region']=='selfchoose':
            #choose a region to score - see if we did this already
            regions = [self._board_state[i,_ST_BDY_SECRET] for i in range(_NUM_REGIONS) if (self._board_state[i,_ST_BDY_SECRET] & pow(2,self._cur_player) > 0)]
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
            self._turn_state[_ST_TN_SCORES:_ST_TN_SCORES+self._num_players]+=final_scores
            self._after_action_step()

    def _score_one_region(self,region,top_only=False):
        assert(region>=0 and region<=_NUM_EXT_REGIONS) 
        cab_counts = self._board_state[_ST_BDX_REGIONS+region,_ST_BDY_CABS : (_ST_BDY_CABS+self._num_players) ]
    
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
                if ranks[k]==1 or (not top_only):
                    final_scores[k]=self._rewards[region-1][ranks[k]-1]
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
        n_cabs = _POWER_CABS[power_id]
        self._board_state[_ST_BDX_PROVINCE,self._cur_player] = self._board_state[_ST_BDX_PROVINCE,self._cur_player] - n_cabs
        self._board_state[_ST_BDX_COURT,self._cur_player] = self._board_state[_ST_BDX_COURT,self._cur_player] + n_cabs
                           
        
    def _setup_caballero_placement(self):
        #set correct state information for where we will be allowed to place caballeros
        self._movement_tracking['from']=[_ST_BDX_COURT]
        self._movement_tracking['lockfrom']=False
        region_king = [b for b in range (_NUM_REGIONS) if self._board_state[b,_ST_BDY_GRANDE_KING] & pow(2,_ST_MASK_KING) != 0][0]
        self._movement_tracking['to']=self._neighbors[region_king]+[_ST_BDX_CASTILLO]
        self._movement_tracking['lockto']=False
        self._movement_tracking['moving'] = True
        card = self._get_current_card()
        n_cabs = min(int(card["name"][4]),self._board_state[_ST_BDX_PROVINCE,self._cur_player])
        pattern = {'player':self._cur_player,'allowed':True,'max':n_cabs,'min':0}
        self._movement_tracking['patterns']=[pattern]
         
    def _init_move_info(self):
        self._movement_tracking = {'from':[],'to':[],'cabs':[],'patterns':[],'queue':[],'player':0,'lockfrom':False,'lockto':False,'moving':False}
        
    def _setup_action(self,alt_action=0):
        #do an instant action, or set up info to enable multi-step actions
        
        self._init_move_info() #wipe out previous info on where caballeros were/were not allowed to move
        card = self._get_current_card()
        action_type = card['actiontype']
        card_details = card.get('details',[])
        if alt_action>0:
            action_type = card_details[alt_action]['type']
            card_details = card_details[alt_action]['details']
                
        #additional setup for cab movement action types
        if action_type == 'all':
            #only one card is an 'and' and it happens to have two move elements
            self._do_caballero_move_info(card_details[0])
            self._add_caballero_move_info(card_details[1])
            multi_step=True
        elif action_type == 'move':
            self._do_caballero_move_info(card_details)
        elif action_type == 'score':
            self._special_score(card_details)

    def _apply_secret_choice(self):
        #figure out what action required a secret choice, and complete it
        card = self._get_current_card()
        action_type = card['actiontype']
        if action_type == 'score':
            self._special_score(card['details'])
        #TODO - more card types


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
        if card_details['numopt']=='lteq':
            pattern['min']=0 
        else:
            pattern['min']=pattern['max']
        
        #there may be a condition on choice of 'from' region - check it
        if card_details['from'].get('condition',None) != None:
            self._movement_tracking['fromcondition']=int(card_details['from']['condition'])
        self._movement_tracking['patterns']=[pattern]   

        #anything that can simply be done, just do it
        fromreg=self._movement_tracking['from']
        toreg = self._movement_tracking['to']
        #assume that 'court' or 'province' only appears by itself
        if (fromreg[0] in [_ST_BDX_COURT,_ST_BDX_PROVINCE]) and (toreg[0] in [_ST_BDX_COURT,_ST_BDX_PROVINCE]):
            assert(len(fromreg)==1 and len(toreg)==1)
            sendCount=card_details['number']
            if sendCount<0:
                sendCount= self._board_state[fromreg[0],_ST_BDY_CABS+self._cur_player] #figure out how much the 'all' in 'send all' is
            for pl in range(self._num_players):
                sendThis=True
                if (card_details['player']=='foreign' and pl==self._cur_player) or  (card_details['player']=='self' and pl!=self._cur_player):
                    sendThis=False
                if sendThis:
                    self._board_state[fromreg[0],_ST_BDY_CABS+self._cur_player] -= sendCount
                    self._board_state[toreg[0],_ST_BDY_CABS+self._cur_player] += sendCount

            #null-out movement info, since we've done the move
            #TODO - in _acard_state set this card from INIT to ACTFIRST or CABFIRST to DONE
            self._init_move_info()
        else:
            self._movement_tracking['moving']=True
    
        
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
            pattern['max']=details['number']
        if details['numopt']=='lteq':
            pattern['min']=0 
        else:
            pattern['min']=pattern['max']        
        self._movement_tracking['patterns'] = self._movement_tracking.get('patterns',[]) + [pattern]

    def _set_valid_cab_movements(self):
        #use movement tracking info to determine which from/to moves are okay
        actions=[]
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
        
        return sorted(actions)

    def _register_cab_moved(self,fromreg,toreg,ofplayer):
        for pattern in self._movement_tracking['patterns']:
            if self._matched_pattern(ofplayer,pattern):
                pattern['cabs']=pattern.get('cabs',0)+1
                return

    def _check_move_patterns_filled(self):
        for pattern in self._movement_tracking['patterns']:
            if pattern['cabs']<pattern['max']:
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
        if valid_action=='move':
            return self._set_valid_cab_movements()
        elif valid_action=='score':
            #if a 'score' action ends up here, we need to do region choice
            for i in range(_NUM_REGIONS):
                actions.append(_ACT_CHOOSE_SECRETS+i)
            return actions
        elif valid_action=='power':
            power_acts = [(_ACT_RETRIEVE_POWERS + i) for i in range(_NUM_POWER_CARDS) if self._past_pcard_state[i] & pow(2,self._cur_player) > 0]
            actions = actions + power_acts
            return actions
        elif valid_action=='grande':
            for i in range(_NUM_REGIONS):
                actions.append(_ACT_MOVE_GRANDES+i)
            return actions
        elif valid_action=='scoreboard':
            for i in range(_NUM_REGIONS*_NUM_SCOREBOARDS):
                actions.append(_ACT_MOVE_SCOREBOARDS+i)
            return actions
        elif valid_action=='king':
            for i in range(_NUM_REGIONS):
                actions.append(_ACT_MOVE_KINGS+i)
            return actions
        elif valid_action=='uniquescore':
            for i in range(_NUM_REGIONS):
                actions.append(_ACT_CHOOSE_SECRETS+i)
            return actions

    def _after_power_choice(self):
        #functions to determine if we should move to the next phase and/or the next player, and who that player might be
        if len(self._playersleft) > 1:
            #set current player
            self._cur_player = self._playersleft[1]
            #move all players up one
            self._playersleft = self._playersleft[1:]
        else:
            #redo the whole queue, and move on to 'action' phase
            powcards = {i:self._pcard_state[i] for i in range(_NUM_POWER_CARDS) if self._pcard_state[i]>0} 
            order=[]
            keys = sorted(powcards.keys(),reverse=True)
            for i in keys:
                player_id = int(np.log2(powcards[i]))
                order = order + [player_id]
            self._cur_player=order[0]
            self._playersleft=order
            self._turn_state[_ST_TN_PHASE]=self._get_phaseid('action')

    def _after_action_step(self):    
        #functions to determine if we should move to the next phase and/or the next player, and who that player might be
        #if still in the process of moving caballeros then don't do anything
        if self._movement_tracking.get('moving',False):
            return
        
        #if we have some cabs to move, ensure we do that
        if self._get_action_card_status()==_ST_AC_PLAY_ACTFIRST:
            self._update_action_card_status(_ST_PHASE_CAB)
            self._setup_caballero_placement()
            return
        
        #if we were moving cabs and now have to do the action card, do that
        if self._get_action_card_status()==_ST_AC_PLAY_CABFIRST:
            self._update_action_card_status(_ST_PHASE_CARD)
            self._setup_action() #TODO - need step here for alt chooser
            return

        #otherwise we're done, update the queue however we need to
        self._update_action_card_status(_ST_PHASE_ACTION)
        if len(self._playersleft) > 1:
            #set current player
            self._cur_player = self._playersleft[1]
            #move all players up one
            self._playersleft = self._playersleft[1:]
        else:
            #redo the whole queue, and move to 'scoring' phase if appropriate, next power choosing phase otherwise
            if _SCORING_ROUND[self._turn_state[_ST_TN_ROUND]]:
                self._turn_state[_ST_TN_PHASE]=_ST_PHASE_SCORE
            else:
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
                self._dealing = True #next state will be "chance" and we will deal cards
                
    def _after_score_step(self):
        new_scores = self._score_all_regions()
        if self._turn_state[_ST_TN_ROUND]==_MAX_TURNS:
            #win points normalised between 0 and 1
            min_score = min(new_scores)
            divisor = max(new_scores)-min_score
            self._win_points = [(f-min_score)/divisor for f in final_scores]
            self._cur_player = pyspiel.PlayerId.TERMINAL
            self._is_terminal=True
        else:
            self._cur_player = self._get_next_player()
        
    # OpenSpiel (PySpiel) API functions are below. These need to be provided by
    # every game. Some not-often-used methods have been omitted.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        elif self._dealing:
            return pyspiel.PlayerId.CHANCE
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
        elif self._dealing:
            return self._deal_actions()
        else:
            actions = []
            if _PHASE_NAMES[self._turn_state[_ST_TN_PHASE]]=='power':
                cards = self._available_powers()
                actions = actions + [c+_ACT_POWERS for c in cards]
            elif _PHASE_NAMES[self._turn_state[_ST_TN_PHASE]]=='action':
                cards = [c for c in range(_NUM_ACTION_CARDS) if self._acard_state[c]==_ST_AC_PLAY_READY]
                actions = actions + [c +_ACT_CARDS for c in cards]
            elif _PHASE_NAMES[self._turn_state[_ST_TN_PHASE]]=='actionchoose':
                actions.append(_ACT_DECIDE_CAB)
                actions.append(_ACT_DECIDE_ACT)
                #TODO - circumstances under which _ACT_DECIDE_ALT will appear
            elif _PHASE_NAMES[self._turn_state[_ST_TN_PHASE]]=='actioncard':
                actions = actions + self._set_valid_actions_from_card() + [_ACT_SKIP]
            elif _PHASE_NAMES[self._turn_state[_ST_TN_PHASE]]=='actioncab':
                actions = actions + self._set_valid_cab_movements() + [_ACT_SKIP]
            else:
                #must be score - choose a secret region
                for i in range(_NUM_REGIONS):
                    actions.append(_ACT_CHOOSE_SECRETS+i)
            
            return actions
        
    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        if not self._dealing:
            raise ValueError("chance_outcomes called on a non-chance state.")
        outcomes = self._deal_actions()
        p = 1.0 / len(outcomes)
        return [(o, p) for o in outcomes]

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
        """Applies the specified action to the state"""

        #possible actions: _ACT_DEAL, _ACT_CARDS (+ _NUM_ACTION_CARDS), _ACT_POWERS (+ _NUM_POWER_CARDS), _ACT_RETRIEVE_POWERS (+ _NUM_POWER_CARDS), 
        # _ACT_DECIDE_CAB, _ACT_DECIDE_ACT = _ACT_DECIDE_CAB + 1, _ACT_CHOOSE_SECRETS (+ _NUM_REGIONS), _ACT_MOVE_GRANDES (+ _NUM_REGIONS), 
        # _ACT_MOVE_KINGS (+ _NUM_REGIONS), _ACT_CAB_MOVES (+ _NUM_CAB_AREAS * _NUM_CAB_AREAS * _MAX_PLAYERS), _ACT_SKIP

        #don't apply an illegal action
        if not action in self.legal_actions():
            return

        if self._dealing:
            self._deal_cards_from_action(action)
        elif action>=_ACT_CARDS and action < _ACT_CARDS + _NUM_ACTION_CARDS:
            self._acard_state[action - _ACT_CARDS] = _ST_AC_PLAY_INIT
            self._turn_state[_ST_TN_PHASE] = _ST_PHASE_CHOOSE
        elif action >= _ACT_POWERS and action < _ACT_POWERS + _NUM_POWER_CARDS:
            self._assign_power(action - _ACT_POWERS)
            self._after_power_choice() #find next player to pick power card, or move on one phase
        elif action >= _ACT_RETRIEVE_POWERS and action < _ACT_RETRIEVE_POWERS + _NUM_POWER_CARDS:
            self._retrieve_power(action - _ACT_RETRIEVE_POWERS)
            self._after_action_step() #check if we need to move to next player, or next step, or keep playing actions
        elif action == _ACT_DECIDE_CAB:
            self._update_action_card_status(_ST_PHASE_CAB)
            self._pack_court()
            self._setup_caballero_placement()
        elif action in [_ACT_DECIDE_ACT,_ACT_DECIDE_ACT_ALT]:
            self._update_action_card_status(_ST_PHASE_CARD)
            self._pack_court()
            self._setup_action(1 if action==_ACT_DECIDE_ACT_ALT else 0)
        elif action >= _ACT_CHOOSE_SECRETS and action < _ACT_CHOOSE_SECRETS + _NUM_REGIONS:
            self._set_secret_region(action - _ACT_CHOOSE_SECRETS)
            if self._turn_state[_ST_TN_PHASE]==_ST_PHASE_SCORE:
                self._after_score_step()
            else:
                #if we weren't chosing for cab movement in scoring, we were choosing for a card action
                self._apply_secret_choice()
                self._after_action_step() 
        elif action >= _ACT_MOVE_GRANDES and action < _ACT_MOVE_GRANDES + _NUM_REGIONS:
            self._move_grande(action - _ACT_MOVE_GRANDES)
            self._after_action_step() 
        elif action >= _ACT_MOVE_KINGS and action < _ACT_MOVE_KINGS + _NUM_REGIONS:
            self._move_king(action - _ACT_MOVE_KINGS)
            self._after_action_step() 
        elif action >= _ACT_MOVE_SCOREBOARDS and action < _ACT_MOVE_SCOREBOARDS + (_NUM_SCOREBOARDS*_NUM_REGIONS):
            board = (action - _ACT_MOVE_SCOREBOARDS)//_NUM_REGIONS
            region = (action - _ACT_MOVE_SCOREBOARDS)%_NUM_REGIONS
            self._move_scoreboard(board,region)
            self._after_action_step() 
        elif action == _ACT_SKIP:
            #whatever we're skipping, the move info goes
            self._init_move_info()
            self._after_action_step()
        else:
            #moving a caballero fromregion, toregion, ofplayer
            fromRegion = (action- _ACT_CAB_MOVES)//(_NUM_CAB_AREAS * _MAX_PLAYERS)
            toRegion = ((action- _ACT_CAB_MOVES)%(_NUM_CAB_AREAS * _MAX_PLAYERS))//_MAX_PLAYERS
            ofPlayer = (action- _ACT_CAB_MOVES)%_MAX_PLAYERS
            self._move_one_cab(fromRegion, toRegion, ofPlayer)        
            self._after_action_step() 
    
    
    def action_to_string(self, arg0, arg1=None):
        """Action -> string. Args either (player, action) or (action)."""
        player = self.current_player() if arg1 is None else arg0
        action = arg0 if arg1 is None else arg1
        actionString=""
        if self._dealing:
            actionString = "Deal " + "|".join([self._cards[self._cardtrack[c]]['name'] for c in self._cards_for_action(action)])
        elif action>=_ACT_CARDS and action < _ACT_CARDS + _NUM_ACTION_CARDS:
            cardname = self._cards[self._cardtrack[action-_ACT_CARDS]]['name']
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
            actionString = "Choose "+ self._regions[action - _ACT_CHOOSE_SECRETS]
        elif action >= _ACT_MOVE_GRANDES and action < _ACT_MOVE_GRANDES + _NUM_REGIONS:
            actionString = "Grande to "+ self._regions[action - _ACT_MOVE_GRANDES]
        elif action >= _ACT_MOVE_KINGS and action < _ACT_MOVE_KINGS + _NUM_REGIONS:
            actionString = "King to "+ self._regions[action - _ACT_MOVE_KINGS]
        elif action >= _ACT_MOVE_SCOREBOARDS and action < _ACT_MOVE_SCOREBOARDS + (_NUM_SCOREBOARDS*_NUM_REGIONS):
            board = (action - _ACT_MOVE_SCOREBOARDS)//_NUM_REGIONS
            region = (action - _ACT_MOVE_SCOREBOARDS)%_NUM_REGIONS
            actionString = "Move scoreboard "+ str(self._scoreboards[board]['points']) +" to " + self._regions[region]   
        elif action == _ACT_SKIP:
            actionString = "Skip this step"
        else:
            #moving a caballero fromregion, toregion, ofplayer
            fromRegion = (action- _ACT_CAB_MOVES)//(_NUM_CAB_AREAS * _MAX_PLAYERS)
            toRegion = ((action- _ACT_CAB_MOVES)%(_NUM_CAB_AREAS * _MAX_PLAYERS))//_MAX_PLAYERS
            ofPlayer = (action- _ACT_CAB_MOVES)%_MAX_PLAYERS
            actionString = self._players[ofPlayer] + " caballero from " + self._regions[fromRegion] + " to " + self._regions[toRegion]        
        
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
        if self._dealing:
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
        #TODO - if game is changed to inherit from pyspiel, this might need changing
        return copy.deepcopy(self)
        


class ElGrandeGame(object):
    """El Grande Game
    """

    def __init__(self):
        pass

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
            chmat = (self._board_state >> channel)%2
            self._obs[channel,:,:]=chmat
        #TODO - check this

    def string_from(self, state, player):
        del player
        return str(state)
