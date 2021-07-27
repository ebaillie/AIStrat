import copy
import pickle
import json

import numpy as np
import random
import couchdb
import pyspiel

_DEFAULT_CONFIG = "el_grande_default" #source of config json if nothing specified on load - file in same directory as .py file
_MAX_PLAYERS = 5
_DEFAULT_PLAYERS = 2
_MAX_REGIONS = 12 # slightly arbitrary - disallow games with "too many" regions
_MAX_CAB_AREAS = _MAX_REGIONS + 3 # cab areas include province, court and castillo
_MAX_ACTION_CARDS = 45 #number of action cards in original El Grande, including vetos
_MAX_POWER_CARDS = 13 #as in original El Grande
_MAX_SCOREBOARDS = 2 #as in original El Grande game

#game phases start,power,action,actionchoose,actioncard1,actioncab1,actioncard2,actioncab2,response,scoring,end
#of these, start,power,action and score can be loaded in from database
_NUM_PHASES = 8 #for calculating possible max game size
_ACTION_TYPES = ['none','move','score','power','grande','scoreboard','king','uniquescore']
_MAX_TURNS = 9 #full game has 9 turns, but we can specify fewer if need be
_SCORING_ROUND = [False,False,False,True,False,False,True,False,False,True]


#Global Board State vars
_ST_MASK_KING = _MAX_PLAYERS #in the king/grande mask, king is at end
_ST_BDY_CABS = 0 #start of cab counts in region columns (per-player)
_ST_BDY_GRANDE_KING = _ST_BDY_CABS + _MAX_PLAYERS # grande/king state (1 bit per player, 1 for the king)
_ST_BDY_SECRET = _ST_BDY_GRANDE_KING + 1 #secret region vote (1 bit per player)
_ST_BDY_END = _ST_BDY_SECRET + _MAX_PLAYERS

#Action card state - int indicators per-card 
_ST_AC_UNPLAYED = 0
_ST_AC_DEALT = 1 
_ST_AC_CHOSEN = 2
_ST_AC_DONE = 3 
_ACT_CARD_STATES = ['unplayed','dealt','chosen','done']
_ACT_CARD_IDS = {'unplayed':_ST_AC_UNPLAYED,'dealt':_ST_AC_DEALT,'chosen':_ST_AC_CHOSEN,'done':_ST_AC_DONE}

#Action card type - 0=instant, 1=1-step, 2=multi-step
#Note 'Angry King' is classed as instant because all the decision steps involved are opponent steps
_ST_STEPS_0 = 0
_ST_STEPS_1 = 1
_ST_STEPS_MULTI = 2

#phases in order(-ish - 'response' can interrupt either actioncard phase)
_ST_PHASE_START=0
_ST_PHASE_POWER=1
_ST_PHASE_ACTION=2
_ST_PHASE_CHOOSE=3
_ST_PHASE_CARD1=4
_ST_PHASE_CAB1=5
_ST_PHASE_CARD2=6
_ST_PHASE_CAB2=7
_ST_PHASE_RESPONSE=8
_ST_PHASE_SCORE=9
_ST_PHASE_END=10
_PHASE_NAMES = ['start','power','action','actionchoose','actioncard1','actioncab1','actioncard2',
                'actioncab2','response','scoring','end']
_PHASE_IDS = {'start':_ST_PHASE_START,'power':_ST_PHASE_POWER,'action':_ST_PHASE_ACTION,'actionchoose':_ST_PHASE_CHOOSE,
                'actioncard1':_ST_PHASE_CARD1,'actioncab1':_ST_PHASE_CAB1,'actioncard2':_ST_PHASE_CARD2,
                'actioncab2':_ST_PHASE_CAB2,'response':_ST_PHASE_RESPONSE,'scoring':_ST_PHASE_SCORE,'end':_ST_PHASE_END}

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
_ACT_POWERS = _ACT_CARDS + _MAX_ACTION_CARDS #select power cards
_ACT_RETRIEVE_POWERS = _ACT_POWERS + _MAX_POWER_CARDS #get back an old power card
_ACT_DECIDE_CAB = _ACT_RETRIEVE_POWERS + _MAX_POWER_CARDS #decide to place cabs first
_ACT_DECIDE_ACT = _ACT_DECIDE_CAB + 1 #decide to play the action card first
_ACT_DECIDE_ACT_ALT = _ACT_DECIDE_ACT + 1 #special for the 'OR' card - decide on the second action
_ACT_CHOOSE_SECRETS = _ACT_DECIDE_ACT_ALT + 1 #choose one secret region
_ACT_MOVE_GRANDES = _ACT_CHOOSE_SECRETS + _MAX_CAB_AREAS
_ACT_MOVE_KINGS = _ACT_MOVE_GRANDES + _MAX_REGIONS
_ACT_MOVE_SCOREBOARDS = _ACT_MOVE_KINGS + _MAX_REGIONS
_ACT_CAB_MOVES = _ACT_MOVE_SCOREBOARDS + (_MAX_SCOREBOARDS*_MAX_REGIONS)
_ACT_SKIP = _ACT_CAB_MOVES + (_MAX_CAB_AREAS * _MAX_CAB_AREAS * _MAX_PLAYERS) #combos of moving a cab from region, to region, of player
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
    max_chance_outcomes=11*11*11*11,  #If Deck counts are 11,11,11,11,1 
    num_players=5,
    min_utility=0.0,
    max_utility=1.0,
    utility_sum=1.0,
    max_game_length=_MAX_PLAYERS*_MAX_TURNS*_NUM_PHASES + 15*_NUM_PHASES + 40) #total phases + cabs out + card sub-actions 

class ElGrandeGameState(pyspiel.State):
    """El Grande Game in open_spiel format
    """

    def __init__(self, game):
        super().__init__(game)
        self._game = game
        self._cur_player = 0
        self._game_step = 0
        self._num_players = game._num_players
        self._is_terminal = False
        self._history = []
        self._winner = False
        self._players = []
        self._end_turn = _MAX_TURNS #default end turn is 9
        self._rsp_player = None #4 vars for dealing with response in move
        self._rsp_phase = None
        self._rsp_steps = []
        self._rsp_finalize = None
        if self._game._game_state != '':
            self._load_game_state(self._game._game_state)
        else:
            #start a game with a random player assortment
            self._players=["P"+str(i) for i in range(self._num_players)]
            self._generate_board({'Players':self._players})

    # Helper functions (not part of the OpenSpiel API).
    
    #info about card names and abilities, region names, player colours
    
    def _get_rid(self,regionName):
        return self._game._region_ids[regionName]
    
    def _get_pid(self,playerName):
        return self._players.index(playerName)
    
    def _get_player_name(self,pid):
        assert(pid<self._num_players)
        return self._players[pid]

    def _get_cid(self,cardName):
        return self._game._cardtrack.index(cardName)

    def _card_info(self,cid):
        cname=self._game._cardtrack[cid]
        cdata=self._game._cards[cname]
        paramtext=[self._game._config_data['units'].get(u,u) for u in cdata.get('textparams',[])]
        return {"name":cname,"deck":cname[:5],"text":cdata['text'].format(*tuple(paramtext))}

    def _get_phaseid(self,phaseName):
        return _PHASE_NAMES.index(phaseName)

    def _get_power_card(self,player=-1):
        if player==-1:
            player = self._cur_player
        cardlist = np.where(self._pcard_state & pow(2,player) == pow(2,player))[0]
        assert(len(cardlist)<=1)
        return cardlist

    def _get_past_power_cards(self,player=-1):
        if player==-1:
            player = self._cur_player
        return np.where(self._past_pcard_state & pow(2,player) == pow(2,player))[0]
    
    def _get_current_card_id(self):
        ccard = [c for c in range(self._game._num_action_cards) if self._acard_state[c] == _ST_AC_CHOSEN]
        assert(len(ccard)<=1) #can have at most one card chosen at a time
        if len(ccard)==0:
            return -1
        else:
            return ccard[0]

    def _get_current_card_name(self):
        cid=self._get_current_card_id()
        if cid<0:
            return ''
        else:
            return self._game._cardtrack[cid]

    def _get_current_card(self):
        cname = self._get_current_card_name()
        if cname=='':
            return None
        else:
            return self._game._cards[cname]

    def _get_round(self):
        return self._turn_state[_ST_TN_ROUND]

    def _get_current_phase_name(self):
        return _PHASE_NAMES[self._turn_state[_ST_TN_PHASE]]

    def _region_has_king(self,region_id):
        return self._board_state[(region_id),_ST_BDY_GRANDE_KING] & (pow(2,_ST_MASK_KING)) == (pow(2,_ST_MASK_KING))
    
    def _king_region(self):
        return int(np.where(self._board_state[:,_ST_BDY_GRANDE_KING]& (pow(2,_ST_MASK_KING))>0)[0][0])
    
    def _region_has_grande(self,region_id,player_id):
        return self._board_state[(region_id),_ST_BDY_GRANDE_KING] & (pow(2,player_id)) == (pow(2,player_id))

    def _grande_region(self,player_id):
        return int(np.where(self._board_state[:,_ST_BDY_GRANDE_KING]& (pow(2,player_id))>0)[0][0])
    
    def _region_cabcount(self,region_id,player_id):
        return int(self._board_state[(region_id),_ST_BDY_CABS + player_id])
    
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
        return int(np.where(self._board_state[:,_ST_BDY_SECRET]& pow(2,player_id) > 0)[0][0])
    
    def _region_presence(self,player,withCastillo=True):
        #in how many regions does this player have any pieces?
        if withCastillo:
            return self._game._num_ext_regions - len(np.where(self._board_state[:self._game._num_ext_regions,player]==0)[0])
        else:
            return self._game._num_regions - len(np.where(self._board_state[:self._game._num_regions,player]==0)[0])

 
    def _board_cabs(self,player):
        #how many total caballeros does this player have on the board?
        return int(sum(self._board_state[:self._game._num_ext_regions,player]))

    def _update_current_card_status(self,status):
        #work out which is the current action card, and set it to new status
        ccard = [c for c in range(self._game._num_action_cards) if self._acard_state[c] == _ST_AC_CHOSEN]
        assert(len(ccard)==1) #raise error if there isn't a card to update
        self._acard_state[ccard[0]] = status
       
    def _card_moves(self):
        # return 0==instant,1=1-step,2=2 or more
        courtstr=self._game._config_data['court']
        provincestr=self._game._config_data['court']
        card = self._get_current_card()
        if card['actiontype']=='move':
            fromreg=card['details']['from']['region']
            toreg=card['details']['to']['region']
            if fromreg=='ownerchooseplus':
                #Angry King
                return _ST_STEPS_0
            if fromreg in [courtstr,provincestr] and toreg in [courtstr,provincestr]:
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
        self._board_state = np.full((self._game._num_cab_areas,_ST_BDY_END),0)
        self._acard_state = np.full(self._game._num_action_cards,0)
        self._acard_round = np.full(self._game._num_action_cards,0)
        if deal:
            self._shuffle_acards() #for deterministic 'deal-at-start' card placement
        self._pcard_state = np.full(self._game._num_power_cards,0)
        self._past_pcard_state = np.full(self._game._num_power_cards,0)
        self._turn_state = np.full(_ST_TN_END,0)
        self._points = self._game._points.copy() 
        self._scoreboards = self._game._scoreboards.copy()
        self._init_move_info()

    def _shuffle_acards(self):
        for i in range(self._game._num_decks):
            deckname='Deck'+str(i+1)
            order = self._game._decks[deckname].copy()
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
        regions = [i for i in range(self._game._num_regions)]
        for i in range(self._num_players):
            region = random.choice(regions)
            self._board_state[region,i]=2
            self._board_state[self._game._court_idx,i]=7
            self._board_state[self._game._province_idx,i]=21
            self._board_state[region,_ST_BDY_GRANDE_KING] |= (pow(2,i))
            regions = [i for i in regions if i!=region]
        king_region = random.choice(regions)
        self._board_state[king_region,_ST_BDY_GRANDE_KING] |= (pow(2,_ST_MASK_KING))
        self._turn_state[_ST_TN_ROUND]=1
        self._turn_state[_ST_TN_PHASE] = _ST_PHASE_POWER 

    #turn all relevant state info from DB format into game format
    def _load_game_state(self,jsonData):
        self._history = jsonData.get('history',[])
        self._blank_board()
        self._state_add_players(jsonData['players'])
        self._state_add_king(jsonData['king'])
        
        #backwards compatibility - importing from TTS version of game unless there's a cardinfo field 
        if jsonData.get('cardinfo',[])==[]:
            self._state_add_cabs_grandes(jsonData['pieces'])
            self._state_add_tts_deck_info(jsonData['cards'],jsonData['pastcards'],jsonData['deckpositions'],jsonData['turninfo'])
            self._state_add_tts_turn_info(jsonData['turninfo'])
        elif jsonData.get('turninfo',[])==[]:
            #no turn info yet - this is initialization
            self._state_add_init_data(jsonData['grandes'],jsonData['cardinfo'])
            self._turn_state[_ST_TN_PHASE]=_ST_PHASE_POWER
            self._turn_state[_ST_TN_ROUND]=1
        else:
            #native python version - compatible with current export
            self._state_add_cabs_grandes(jsonData['pieces'])
            self._state_add_card_info(jsonData['cardinfo'],jsonData['turninfo']['round'])
            self._state_add_turn_info(jsonData['turninfo'])
            self._state_add_point_info(jsonData['pointinfo'])

    def _json_for_game_state(self):
        jsonData={}
        jsonData['history']=self._history.copy()
        jsonData['players']=self._json_for_players()
        jsonData['king']=self._json_for_king()
        jsonData['pieces']=self._json_for_pieces()
        jsonData['cardinfo']=self._json_for_card_info()
        jsonData['turninfo']=self._json_for_turn_info()
        jsonData['pointinfo']=self._json_for_point_info()
        jsonData['config']=self._game._config_file
        return jsonData
 
    def _state_add_players(self, playerData):
        self._players = playerData
        self._num_players = len(self._players)
        self._win_points = np.full(self._num_players, 0)
        self._state_returns = np.full(self._num_players, 0)
        self._playersleft = [self._get_pid(p) for p in self._players]
        self._playersdone = []

    def _json_for_players(self):
        return self._players
 
    def _state_add_king(self,region_name):
        region_id = self._get_rid(region_name)
        assert(region_id < self._game._castillo_idx )
        self._board_state[region_id,_ST_BDY_GRANDE_KING] |= (pow(2,_ST_MASK_KING))

    def _json_for_king(self):
        return self._game._regions[self._king_region()]
   
    def _state_add_init_data(self,grandes,cardrounds):
        #put cabs and grandes down in known regions
        for pl in grandes:
            i=self._get_pid(pl)
            region = self._get_rid(grandes[pl])
            self._board_state[region,i]=2
            self._board_state[self._game._court_idx,i]=7
            self._board_state[self._game._province_idx,i]=21
            self._board_state[region,_ST_BDY_GRANDE_KING] |= (pow(2,i))
        #determine order of cards
        self._acard_round=np.array(cardrounds)
        self._acard_state = np.array([_ST_AC_DEALT if self._acard_round[r]==1 else _ST_AC_UNPLAYED for r in range(len(self._acard_round))])

    def _state_add_cabs_grandes(self,data):
        for player_name in data.keys():
            player_id = self._get_pid(player_name)
            for key in data[player_name].keys():
                if key=="grande":
                    region_id = self._get_rid(data[player_name][key])
                    assert(region_id < self._game._castillo_idx)
                    self._board_state[region_id,_ST_BDY_GRANDE_KING] |= (pow(2,player_id))
                else:
                    region_id = self._get_rid(key)
                    assert(region_id < self._game._num_cab_areas)
                    self._board_state[region_id,_ST_BDY_CABS + player_id]=data[player_name][key]           
   
    def _json_for_pieces(self):
        jsonData={}
        for p in range(self._num_players):
            pieceData={}
            for r in range(self._game._num_cab_areas):
                n=int(self._region_cabcount(r,p))
                if n>0:
                    pieceData[self._game._regions[r]]=n
            pieceData['grande']=self._game._regions[self._grande_region(p)]
            jsonData[self._players[p]]=pieceData
        return jsonData

    def _state_add_card_info(self,jsonData,thisRound):
        #self._acard_state = np.array([_ACT_CARD_IDS[a] for a in jsonData['action_cards']])
        self._acard_round = np.array(jsonData['round_cards'])
        self._acard_state = np.array([_ST_AC_UNPLAYED if self._acard_round[r]>thisRound else _ST_AC_DONE for r in range(len(self._acard_round))])
        for c in jsonData['action_cards']:
            self._acard_state[int(c)]=_ACT_CARD_IDS[jsonData['action_cards'][c]['state']]
        self._pcard_state = np.full(self._game._num_power_cards,0)
        self._past_pcard_state = np.full(self._game._num_power_cards,0)
        for pl in jsonData['power_cards']:
            self._pcard_state[jsonData['power_cards'][pl]-1] |= pow(2,self._get_pid(pl))
        for pl in jsonData['past_power']:
            for pcard in jsonData['past_power'][pl]:
                self._past_pcard_state[pcard-1] |= pow(2,self._get_pid(pl))

    def _json_for_card_info(self):
        #represent card states as strings, round info as int round numbers
        jsonData={}
        #make sure we always have the last card - "Deck5" - listed
        numCards=len(self._acard_state)
        jsonData['action_cards']={str(r):{"state":_ACT_CARD_STATES[self._acard_state[r]],
            "text":self._card_info(r)["text"],
            "deck":self._card_info(r)["deck"]} 
            for r in range(numCards) if r==(numCards-1) or self._acard_round[r]==self._get_round()}
         
        jsonData['round_cards']=self._acard_round.tolist()
        jsonData['power_cards']={}
        jsonData['past_power']={}
        for p in range(self._num_players):
            cur_card = self._get_power_card(p)
            if len(cur_card)==1:
                jsonData['power_cards'][self._players[p]]=int(cur_card[0]+1)
            jsonData['past_power'][self._players[p]] = [int(pc+1) for pc in self._get_past_power_cards(p)] 
        return jsonData
                                                                              
    def _set_rewards(self,new_scores):
        #support both TERMINAL and REWARD mode
        self._turn_state[_ST_TN_SCORES:_ST_TN_SCORES+self._num_players]+=new_scores
        self._state_returns = new_scores - np.mean(new_scores)

    def _current_score(self):
        return self._turn_state[_ST_TN_SCORES:_ST_TN_SCORES+self._num_players]

    def _next_score_step_player(self):
        #find another player who needs to make a secret region choice in scoring round
        #return -1 if no such player
        waiting_players = [i for i in range(self._num_players) if not self._has_secret_region(i) and self._region_cabcount(self._game._castillo_idx,i)>0]
        if len(waiting_players)>0:
            return waiting_players[0]
        else:
            return -1

    def _state_add_tts_deck_info(self,cards,pastcards,deckpositions,data):
        #action cards, sorted by deck
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
        self._turn_state[_ST_TN_ROUND]=data['round']
        self._turn_state[_ST_TN_PHASE]=self._get_phaseid(data['phase'])
        for player_name in data['scores'].keys():
            player_id = self._get_pid(player_name)
            score = data['scores'][player_name]
            self._turn_state[_ST_TN_SCORES+player_id]=score
        self._playersleft=[self._get_pid(p) for p in data['playersleft']]
        self._playersdone=[self._get_pid(p) for p in data['playersdone']]
        self._cur_player = self._playersleft[0] if len(self._playersleft)>0 else 0
        for pl in data['secretchoice']:
            self._set_secret_region(self._get_rid(data['secretchoice'][pl]),self._get_pid(pl))
        self._game_step = data.get('gamestep',0)
        self._movement_tracking = data['moveinfo']
        if data.get('rspinfo',[])!=[]:
            self._rsp_player = data['rspinfo']['player']
            self._rsp_phase = self._get_phaseid(data['rspinfo']['phase'])
            self._rsp_steps = data['rspinfo']['steps'].copy()
            assert(len(self._rsp_steps)>0)
            self._cur_player = self._rsp_steps[0]['player']
            self._rsp_finalize = data['rspinfo']['finalize']

    def _json_for_turn_info(self):
        jsonData={}
        jsonData['round']=int(self._get_round())
        jsonData['phase']=self._get_current_phase_name()
        cscores=self._current_score()
        jsonData['scores']={self._get_player_name(p):int(cscores[p]) for p in range(self._num_players)}
        jsonData['playersdone']=[self._get_player_name(p) for p in self._playersdone]
        jsonData['playersleft']=[self._get_player_name(p) for p in self._playersleft]
        jsonData['secretchoice']={self._get_player_name(p):self._game._regions[self._secret_region(p)] for p in range(self._num_players) if self._has_secret_region(p)}
        jsonData['gamestep']=self._game_step
        jsonData['moveinfo']=self._movement_tracking
        if self._turn_state[_ST_TN_PHASE]==_ST_PHASE_RESPONSE:
            jsonData['rspinfo']={"player":self._rsp_player,"phase":_PHASE_NAMES[self._rsp_phase],"steps":self._rsp_steps,"finalize":self._rsp_finalize}
        return jsonData

    def _state_add_point_info(self,data):
        self._points = {self._get_rid(r):data[r] for r in data}

    def _json_for_point_info(self):
        jsonData={self._game._regions[r]:self._points[r] for r in range(len(self._points))}
        return jsonData

    def _state_add_tts_turn_info(self,data):
        #power cards
        if len(data['powercards'])>0:
            for player_name in data['powercards'].keys():
                player_id = self._get_pid(player_name)
                power_id = int(data['powercards'][player_name]) 
                assert((power_id-1) <= self._game._num_power_cards and power_id > 0) #power_id from 1 to self._game._num_power_cards
                self._pcard_state[(power_id-1)] |= pow(2,player_id)

        #past power cards
        if len(data['powerplayed'])>0:
            for player_name in data['powerplayed'].keys():
                player_id = self._get_pid(player_name)
                for power_id in data['powerplayed'][player_name]:
                    assert((power_id-1) <= self._game._num_power_cards and power_id > 0) #power_id from 1 to self._game._num_power_cards
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
            self._acard_state = np.array([1 if self._acard_round[i]==self._turn_state[_ST_TN_ROUND] else self._acard_state[i] for i in range(self._game._num_action_cards)])
            self._acard_state[-1]=1
            self._playersleft=[self._get_pid(p) for p in data['playersleft']]
            self._playersdone=[self._get_pid(p) for p in data['playersdone']]
            self._cur_player = self._playersleft[0]
        else:
            self._playersleft=[self._get_pid(p) for p in data['playersleft']]
            self._playersdone=[self._get_pid(p) for p in data['playersdone']]
            self._cur_player = self._playersleft[0] if len(self._playersleft)>0 else 0
            

    #functions for doing actions
    
    def _deal_actions(self):
        deck_ends=[len(deck) for deck in self._game._decks]
        dlists = []
        for deck in self._game._decks:
            deck_id=int(deck[4]) #5th character of 'Deckn'
            cards = [i for i in range(deck_ends[deck_id-1])  if self._acard_state[self._get_cid(self._game._decks[deck][i])]==_ST_AC_UNPLAYED]
            dlists = dlists+[cards]
        action_list = [a + deck_ends[0]*(b + deck_ends[1]*(c + deck_ends[2]*d)) for a in dlists[0] for b in dlists[1] for c in dlists[2] for d in dlists[3]]
        return sorted(action_list)

    def _deck_pos_for_action(self,action):
        deck_ends=[len(deck) for deck in self._game._decks]
        deck1 = action%deck_ends[0]
        deck2 = (action%(deck_ends[0] * deck_ends[1]))//deck_ends[0]
        deck3 = (action%(deck_ends[0] * deck_ends[1] * deck_ends[2]))//(deck_ends[0] * deck_ends[1])
        deck4 = action //(deck_ends[0] * deck_ends[1] * deck_ends[2])
        deck5 = 0 # single deck5 card is always chosen
        return (deck1,deck2,deck3,deck4,deck5)
    
    def _deal_cards_from_action(self,action):
        #mark previous cards as played
        self._acard_state = [_ST_AC_UNPLAYED if c==_ST_AC_UNPLAYED else _ST_AC_DONE for c in self._acard_state]
        deck_positions = self._deck_pos_for_action(action)
        for i in range(len(deck_positions)):
            deck="Deck"+str(i+1)
            self._acard_state[self._get_cid(self._game._decks[deck_positions[i]])]=_ST_AC_DEALT
        self._turn_state[_ST_TN_PHASE]=self._get_phaseid('power')
        self._dealing=False
             
    def _cards_for_action(self,action):
        deck_positions = self._deck_pos_for_action(action)
        guids = [self._game._decks["Deck"+str(i+1)][deck_positions[i]] for i in range(len(deck_positions))]
        return guids
             
    def _assign_power(self,power_id,player_id=-1):
        #power_id is array position 0..12
        if player_id==-1:
            player_id = self._cur_player
        self._pcard_state[power_id] |= pow(2,player_id)
        self._past_pcard_state[power_id] |= pow(2,player_id)
        
    def _retrieve_power(self,power_id):
        self._past_pcard_state[power_id] -= pow(2,self._cur_player)
       
    def _available_powers(self):
        #power cards available to current player in array position 0..12
        #check everyone's current cards, and current player's past 
        in_deck = [p for p in range(self._game._num_power_cards) if self._pcard_state[p]==0]
        unused = [p for p in range(self._game._num_power_cards) if self._past_pcard_state[p] & pow(2,self._cur_player)==0]
        return [p for p in range(self._game._num_power_cards) if p in in_deck and p in unused]

    def _set_secret_region(self,region_id,player_id=-1):
        if player_id==-1:
            player_id = self._cur_player
        #ensure only one set at a time
        mask = pow(2,(_MAX_PLAYERS+1))-pow(2,player_id)-1 #masks out everything except this player
        self._board_state[:self._game._court_idx,_ST_BDY_SECRET] &= np.full(self._game._court_idx,mask)
        self._board_state[region_id,_ST_BDY_SECRET] |= pow(2,player_id)

    def _move_grande(self,region_id):
        #mask out current player's grande from everywhere else
        #note the mask for this has to fit one entry per grande for all players, plus also king
        mask = pow(2,(_MAX_PLAYERS+2))-pow(2,self._cur_player)-1 #masks out everything except this player
        self._board_state[:self._game._court_idx,_ST_BDY_GRANDE_KING] &= np.full(self._game._court_idx,mask)
        self._board_state[region_id,_ST_BDY_GRANDE_KING] |= pow(2,self._cur_player)
        
    def _move_king(self,region_id):
        mask = pow(2,(_MAX_PLAYERS+2))-pow(2,_ST_MASK_KING)-1 #masks out everything except king
        self._board_state[:self._game._court_idx,_ST_BDY_GRANDE_KING] &= np.full(self._game._court_idx,mask)
        self._board_state[region_id,_ST_BDY_GRANDE_KING] |= pow(2,_ST_MASK_KING)

    def _move_scoreboard(self,board_id,region_id):
        points = self._scoreboards[board_id]['points']
        self._points[region_id]=points
        oldregion = self._scoreboards[board_id].get('region',-1)
        if oldregion>=0 and oldregion<len(self._points):
            self._points[oldregion] = self._game._points[oldregion]
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
        
    def _quickmove_cabs(self, from_region, to_region, num=-1, of_player=-1):
        #shift cabs about without doing any registration or tracking
        if of_player<0:
            of_player=self._cur_player
        if num<0:
            num=self._board_state[from_region,of_player]
        else:
            num=min(num,self._board_state[from_region,of_player])
        self._board_state[from_region,of_player] = self._board_state[from_region,of_player] - num
        self._board_state[to_region,of_player] = self._board_state[to_region,of_player] + num
 
    def _move_castillo_pieces(self):
        for i in range(self._num_players):
            if self._has_secret_region(i):
                region = self._secret_region(i)
                self._quickmove_cabs(self._game._castillo_idx,region,-1,i)
    
    def _region_str(self,region_id):
        retstr = "|".join([str(i) for i in self._board_state[region_id,_ST_BDY_CABS:(_ST_BDY_CABS + self._num_players)]])
        grandes = [str(i) for i in range(self._num_players) if self._region_has_grande(region_id,i)]
        if len(grandes)>0:
            retstr = retstr + " G({0})".format("|".join(grandes))
        if self._region_has_king(region_id):
            retstr = retstr + " K"
        retstr = retstr.ljust(18," ")+ "-  " + self._game._regions[region_id].rjust(18," ")
        if region_id < self._game._num_ext_regions:
            retstr = retstr + str(self._points[region_id])
        return retstr
    
    def _power_str(self,player_id):
        current = self._get_power_card(player_id) 
        if len(current)==0:
            cstr = "X"
        else:
            cstr = str(current[0] + 1)
        past = [str(p+1) for p in np.where(self._past_pcard_state & pow(2,player_id) == pow(2,player_id))[0]]
        return "{0} [{1}] - player {2} ({3}))".format(cstr,(",".join(past)),player_id,self._get_player_name(player_id))

    def _acard_str(self,cardpos):
        cardstr = self._game._cardtrack[cardpos]
        if self._acard_state[cardpos]==_ST_AC_DEALT:
            return cardstr
        else:
            return '*{0}*'.format(cardstr)
 
    def _action_str(self):
        activecards = [c for c in range(self._game._num_action_cards) if self._acard_state[c]!=_ST_AC_UNPLAYED and self._acard_state[c]!= _ST_AC_DONE]
        assert(len(activecards)<=self._game._num_decks)
        names = [self._acard_str(c) for c in activecards]
        return "|".join(names)
    
    def _turn_info_str(self):
        return "[{0}] - Round {1} {2} (player {3})\n Cumulative scores: {4}".format(self._game_step,self._get_round(),
            self._get_current_phase_name(),self._cur_player,self._current_score())    
   
    def _unique_score(self):
        #determine any doubles before scoring
        regions=[]
        for p in range(self._num_players):
            regions.append(self._secret_region(p))
        score_regions=[]
        for i in range(self._num_players):
            if not (regions[i] in (regions[:i]+regions[i+1:])):
                score_regions.append(regions[i])
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
            for i in range(self._game._num_ext_regions): 
                if self._points[i][0]==4:
                    final_scores = final_scores + self._score_one_region(i)
        elif region=='fives':
            for i in range(self._game._num_ext_regions):
                if self._points[i][0]==5:
                    final_scores = final_scores + self._score_one_region(i)
        elif region=='sixsevens':
            for i in range(self._game._num_ext_regions):
                if self._points[i][0]==6 or self._points[i][0]==7:
                    final_scores = final_scores + self._score_one_region(i)
        elif region=='castillo':
            final_scores = final_scores + self._score_one_region(self._game._castillo_idx)
        elif region=='locab':
            loregions=[]
            currentlow=100
            for i in range(self._game._num_ext_regions):
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
            for i in range(self._game._num_ext_regions):
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
            regions = [i for i in range(self._game._num_regions) if self._region_is_secret_choice(i)]
            assert(len(regions)<=1)
            if len(regions)==0:
                choice_step=True
            else:
                final_scores = final_scores + self._score_one_region(regions[0])
        else:
            #last possibility is the one where we just score the top in all regions
            for i in range(self._game._num_ext_regions):
                final_scores = final_scores + self._score_one_region(i,True)
                
        if not choice_step:
            self._set_rewards(final_scores)

    def _rank_region(self, region):
        assert(region>=0 and region<self._game._num_ext_regions) 
        cab_counts = self._board_state[region,_ST_BDY_CABS : (_ST_BDY_CABS+self._num_players) ]
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
        assert(region>=0 and region<self._game._num_ext_regions) 
        cab_counts = self._board_state[region,_ST_BDY_CABS : (_ST_BDY_CABS+self._num_players) ]
    
        #rank and score the counts
        ranks = self._rank_region(region)
        #score for all players
        final_scores=np.full(self._num_players,0)
        for k in ranks.keys():
            if ranks[k]>0 and ranks[k]<=3:
                if ranks[k]==1 or (not top_only):
                    final_scores[k]=self._points[region][ranks[k]-1]
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
        for r in range(self._game._num_ext_regions):
            final_scores = final_scores+self._score_one_region(r)
        return final_scores
   
    def _scores_as_margins(self,player_scores):
        #represent all players' scores in margin format (distance from the midpoint between first and second)
        fscore=max(player_scores)
        sscore=0
        if len(player_scores)>1: #always true except in test games
            sscore = sorted(player_scores,reverse=True)[1]
        scorepoint=(fscore+sscore)/2
        #players are scored relative to the midpoint between first and second place
        return player_scores - scorepoint
 
    def _pack_court(self):
        #move the correct number of caballeros from province to court, at the point where player action commences
        power_id = np.where(self._pcard_state & pow(2,self._cur_player) == pow(2,self._cur_player))[0][0]
        self._quickmove_cabs(self._game._province_idx,self._game._court_idx,self._game._power_cabs[power_id])
                           
        
    def _setup_caballero_placement(self):
        #set correct state information for where we will be allowed to place caballeros
        self._movement_tracking['from']=[self._game._court_idx]
        self._movement_tracking['lockfrom']=False
        self._movement_tracking['to']=self._game._neighbors[self._king_region()]+[self._game._castillo_idx]
        self._movement_tracking['lockto']=False
        self._movement_tracking['moving'] = True
        card_name = self._get_current_card_name()
        n_cabs = min(int(card_name[4]),self._region_cabcount(self._game._court_idx,self._cur_player))
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
            self._setup_response_vars(0,'uniquescore','secret')
        elif action_type == 'move':
            #Province/Provinceall/Eviction
            fromreg = card['details']['from']['region']
            if fromreg=='ownerchoose':
                fnl = 'alltoprovince' if card['details']['from']['condition']=='1' else 'twotoprovince'
                self._setup_response_vars(card['details']['from']['condition'],fnl,'secret')
            else:
                self._setup_response_vars(0,'toregion','secret')


    def _do_caballero_move_info(self,card_details):  
        #make movable caballeros interactable
        #'from' values are court, or region of your choice
        #self._movement_tracking = {['from']={},['to']={},['patterns']={},
        #                     ['player']=None,['lockfrom']=false,['lockto']=false,['moving']=false}

        self._movement_tracking['player']=self._cur_player
        for v in ['from','to']:
            if card_details[v]['region'] =='court':
                self._movement_tracking[v]=[self._game._court_idx]
            elif card_details[v]['region'] =='province':
                self._movement_tracking[v]=[self._game._province_idx]
            elif card_details[v]['region']=='selfchoose':
                #current King's region shouldn't be in the list
                the_regions=[i for i in range(self._game._num_regions) if not self._region_has_king(i)]
                if v=='to':
                    the_regions = the_regions + [self._game._castillo_idx]
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
            #Setup move info and change player to responder
            self._setup_response_vars(int(card_details['number']),'','move')
        else:
            #'Decay','Decayall' and 'Court' 
            sendCount=card_details['number']
            for pl in range(self._num_players):
                sendThis=True
                if (card_details['player']=='foreign' and pl==self._cur_player) or  (card_details['player']=='self' and pl!=self._cur_player):
                    sendThis=False
                if sendThis:
                    self._quickmove_cabs(fromreg[0],toreg[0],sendCount,pl)

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


    def _setup_response_vars(self,condition,finalStep,acttype):
        #set response data so that opponents can make secret choices
        self._rsp_player = self._cur_player
        self._rsp_phase =  self._turn_state[_ST_TN_PHASE]
        self._rsp_steps = []
        self._rsp_finalize = finalStep
        #not for secret choice needed from current player
        players_to_choose = [(i+self._cur_player)%self._num_players for i in range(1,self._num_players)]
        assert(len(players_to_choose)>0)
        if acttype=='move':
            #'Angry King' - each player moves caballeros back to province in order
            for p in players_to_choose:
                for i in range(condition):
                    self._rsp_steps.append({'player':p,'court':True,'cabs':1,'do':'toprovince'})
        else:
            # Province/Provinceall/Eviction/unique score - each player makes one secret choice
            for p in players_to_choose:
                self._rsp_steps.append({'player':p,'court':False,'cabs':int(condition),'do':'setsecret'})
        self._cur_player = self._rsp_steps[0]['player']
        self._turn_state[_ST_TN_PHASE]=_ST_PHASE_RESPONSE

        self._init_move_info()

    def _king_placement_details(self,card_details):
        if card_details['condition']=='all':
            return [_ACT_MOVE_KINGS+i for i in range(self._game._num_regions)]
        else:
            return [_ACT_MOVE_KINGS+i for i in range(self._game._num_regions) if i in self._game._neighbors[self._king_region()]]

    def _set_next_response_vars(self):
        #move on to next thing in response
        self._rsp_steps.pop(0)
        if len(self._rsp_steps)>0:
            self._cur_player=self._rsp_steps[0]['player']
        else:
            self._finalize_response()
            self._cur_player = self._rsp_player
            self._turn_state[_ST_TN_PHASE]=self._rsp_phase
            self._rsp_player = None
            self._rsp_phase = None
            self._rsp_finalize = None

    def _finalize_response(self):
        #finalize for Province/ProvinceAll/Eviction/Deck4_Special 
        if self._rsp_finalize=='alltoprovince':
            #send everything from selected regions to province
            for p in range(self._num_players):
                region=self._secret_region(p)
                self._quickmove_cabs(region,self._game._province_idx,-1,p)
        if self._rsp_finalize=='twotoprovince':
            #send two from selected regions to province
            for p in range(self._num_players):
                region=self._secret_region(p)
                self._quickmove_cabs(region,self._game._province_idx,2,p)
        elif self._rsp_finalize=='toregion':
            #send everything from region of rsp player to others' region choices
            fromreg = self._secret_region(self._rsp_player)
            for p in range(self._num_players):
                toreg = self._secret_region(p)
                self._quickmove_cabs(fromreg,toreg,-1,p)
        elif self._rsp_finalize=='uniquescore':
            self._unique_score()

    def _set_valid_cab_movements(self,fromcard=True):
        #use movement tracking info to determine which from/to moves are okay
        actions=[]
        #if secret choice is involved, interrupt to do this
        ccard=self._get_current_card()
        if fromcard and ccard['actiontype']=='move':
            checkregionfrom = ccard['details']['from']['region']
            checkregionto = ccard['details']['to']['region']
            if checkregionfrom =='ownerchoose' or checkregionto == 'ownerchoose':
                actions = [i + _ACT_CHOOSE_SECRETS for i in range(self._game._num_regions) if self._board_state[i,self._cur_player]>=self._movement_tracking.get('fromcondition',0) and not self._region_has_king(i)]
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
                    if self._board_state[fromreg,player] >0 and fromreg!=toreg:
                        #there is a caballero here of the correct colour, so this move action is okay
                        #don't list moves from/to the same position
                        actions.append(_ACT_CAB_MOVES + player + _MAX_PLAYERS*(toreg + self._game._num_cab_areas*fromreg))
        
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
       
    def _get_region_actions_for_rsp_condition(self):
        #get from the first condition in the response list. Include safeties for secret choice if we would end with
        #no action choices
        courtact = []
        if self._rsp_steps[0]['court']:
            courtact= [_ACT_CHOOSE_SECRETS + self._game._court_idx] 

        cabcondition=self._rsp_steps[0]['cabs']
        actions=[]
        while actions==[] and cabcondition>=0:
            actions = [(i + _ACT_CHOOSE_SECRETS) for i in range (self._game._num_regions) if 
                self._region_cabcount(i,self._rsp_steps[0]['player'])>=cabcondition and not 
                self._region_has_king(i)] + courtact
            cabcondition -=1

        return actions
         
        
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
            actions = actions + [(i + _ACT_CHOOSE_SECRETS) for i in range(self._game._num_regions) if not self._region_has_king(i)]
            return actions
        elif valid_action=='choose':
            #check if there is a movement pattern - if so, follow it, if not, ask whether we're on 1 or 2
            if self._movement_tracking['moving']:
                return self._set_valid_cab_movements()
            else:
                return[_ACT_DECIDE_ACT,_ACT_DECIDE_ACT_ALT]
        elif valid_action=='power':
            actions = actions + [(_ACT_RETRIEVE_POWERS + i) for i in range(self._game._num_power_cards) if self._past_pcard_state[i] & pow(2,self._cur_player) > 0]
            return actions
        elif valid_action=='grande':
            actions = actions + [(i + _ACT_MOVE_GRANDES) for i in range(self._game._num_regions) if not self._region_has_king(i)]
            return actions
        elif valid_action=='scoreboard':
            actions = actions + [(i+(j*self._game._num_regions) + _ACT_MOVE_SCOREBOARDS) for i in range(self._game._num_regions) for j in range(self._game._num_scoreboards) if not self._region_has_king(i)]
            return sorted(actions)


    def _after_power_choice(self):
        #functions to determine if we should move to the next phase and/or the next player, and who that player might be
        if len(self._playersleft) > 1:
            #set current player
            self._cur_player = self._playersleft[1]
            #move all players up one
            self._playersdone = self._playersdone + self._playersleft[:1] 
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
       
        #if it's in the middle of an opponent response, also don't do anything
        if self._turn_state[_ST_TN_PHASE]==_ST_PHASE_RESPONSE:
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
            self._playersdone = self._playersdone + self._playersleft[:1]
            self._playersleft = self._playersleft[1:]
            self._turn_state[_ST_TN_PHASE]=_ST_PHASE_ACTION
        else:
            #redo the whole queue, and move to 'scoring' phase if appropriate, 
            #next power choosing phase otherwise
            if _SCORING_ROUND[self._turn_state[_ST_TN_ROUND]]:
                self._turn_state[_ST_TN_PHASE]=_ST_PHASE_SCORE
                #make sure secret region choices are all blank before we start this
                self._board_state[:,_ST_BDY_SECRET]=0
                #players with no cabs in castillo are done, other players are left
                self._playersdone=[i for i in range(self._num_players) if self._region_cabcount(self._game._castillo_idx,i)==0]
                self._playersleft=[i for i in range(self._num_players) if self._region_cabcount(self._game._castillo_idx,i)>0]
                if len(self._playersleft)>0:
                    self._cur_player=self._playersleft[0]
                else:
                    self._after_score_step()
            else:
                self._update_players_after_action()
                
    def _after_score_step(self):
        #set next player if necessary
        self._playersdone=self._playersdone+self._playersleft[:1]
        self._playersleft=self._playersleft[1:]
        if len(self._playersleft)>0:
            self._cur_player=self._playersleft[0]
            return

        #if everybody who needs to has chosen a secret region, do the scoring
        #score the castillo, move castillo pieces out, then score everything
        new_scores=self._score_one_region(self._game._castillo_idx)
        self._move_castillo_pieces()
        new_scores+=self._score_all_regions()
        self._set_rewards(new_scores)
        final_scores = self._current_score()
        self._board_state[:,_ST_BDY_SECRET]=0 
        if self._turn_state[_ST_TN_ROUND]==self._end_turn:
            # turn scores into win points
            self._win_points = self._scores_as_margins(final_scores)
            self._cur_player = pyspiel.PlayerId.TERMINAL
            self._is_terminal=True
        else:
            self._update_players_after_action()

    def _update_players_after_power(self):
        powcards = {i:self._pcard_state[i] for i in range(self._game._num_power_cards) if self._pcard_state[i]>0} 
        order=[]
        keys = sorted(powcards.keys(),reverse=True)
        for i in keys:
            player_id = int(np.log2(powcards[i]))
            order = order + [player_id]
        self._cur_player=order[0]
        self._playersleft=order
        self._playersdone=[]
        self._turn_state[_ST_TN_PHASE]=self._get_phaseid('action')
        
    def _update_players_after_action(self):
        #get everything ready for next round
        powcards = {i:self._pcard_state[i] for i in range(self._game._num_power_cards) if self._pcard_state[i]>0} 
        lowest = sorted(powcards.keys())[0]
        start_player = int(np.log2(powcards[lowest]))
        self._cur_player = start_player
        order = [start_player]
        for i in range(1,self._num_players):
            order = order +[(start_player+i) % self._num_players]
        self._playersleft = order
        self._playersdone = []
        self._turn_state[_ST_TN_PHASE]=_ST_PHASE_POWER
        self._turn_state[_ST_TN_ROUND]+=1
        self._pcard_state = np.full(self._game._num_power_cards,0)
        #make the right cards dealt
        round = self._turn_state[_ST_TN_ROUND]
        for i in range(self._game._num_action_cards):
            if self._acard_round[i]==round:
                self._acard_state[i] = _ST_AC_DEALT
            elif self._acard_round[i]==(round-1):
                self._acard_state[i] = _ST_AC_DONE
        self._acard_state[-1] = _ST_AC_DEALT

    def scoring_order(self,player=-1):
        #order in which players will do their (secret) castillo choice - needed for castillo subgame
        if player<0:
            player=self._cur_player
        so=self._playersleft+self._playersdone
        zero_point=so.index(player)
        new_idxs={v:(so.index(v)-zero_point)%len(so) for v in so}
        return new_idxs

    def castillo_game_string(self,player=-1):
        #translate game state into CastilloGame format, for running tiny sims
        state_vals={"players":self._num_players,"rewards":self._points,"king":self._king_region()+1}
        #put player state into castillo game state in order, starting from current player
        #region 0 is the castillo
        board=np.full(self._num_players*self._game._num_ext_regions,0)
        grandes=np.full(self._num_players,0)
        scores=np.full(self._num_players,0)
        new_idxs=self.scoring_order(player)
        for p in new_idxs:
            idx=new_idxs[p]
            for r in range(self._game._num_regions):
                board[(r+1)*self._num_players+idx] = self._board_state[r,p]    
            #castillo
            board[idx] = self._board_state[self._game._castillo_idx,p]
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
        else:
            actions = []
            if self._turn_state[_ST_TN_PHASE]==_ST_PHASE_POWER:
                cards = self._available_powers()
                actions = actions + [c+_ACT_POWERS for c in cards]
            elif self._turn_state[_ST_TN_PHASE]==_ST_PHASE_ACTION:
                cards = [c for c in range(self._game._num_action_cards) if self._acard_state[c]==_ST_AC_DEALT]
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
            elif self._turn_state[_ST_TN_PHASE] == _ST_PHASE_RESPONSE:
                actions = self._get_region_actions_for_rsp_condition()
            else:
                #must be score - choose a secret region (not one with the King)
                actions = actions + [(i + _ACT_CHOOSE_SECRETS) for i in range (self._game._num_regions) if not self._region_has_king(i)]
            
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

        #possible actions: _ACT_DEAL, _ACT_CARDS (+ _NUM_ACTION_CARDS), _ACT_POWERS (+ _NUM_POWER_CARDS), _ACT_RETRIEVE_POWERS (+ _NUM_POWER_CARDS), 
        # _ACT_DECIDE_CAB, _ACT_DECIDE_ACT = _ACT_DECIDE_CAB + 1, _ACT_CHOOSE_SECRETS (+ _NUM_CAB_AREAS), _ACT_MOVE_GRANDES (+ _NUM_REGIONS), 
        # _ACT_MOVE_KINGS (+ _NUM_REGIONS), _ACT_CAB_MOVES (+ _NUM_CAB_AREAS * _NUM_CAB_AREAS * _MAX_PLAYERS), _ACT_SKIP

        #don't apply an illegal action
        if not action in self.legal_actions():
            return

        #we're going to do this, so record it in the history
        self._history.append(action)
        self._set_rewards(np.full(self._num_players,0))
        self._game_step+=1

        if action>=_ACT_CARDS and action < _ACT_CARDS + self._game._num_action_cards:
            self._acard_state[action - _ACT_CARDS] = _ST_AC_CHOSEN
            self._turn_state[_ST_TN_PHASE] = _ST_PHASE_CHOOSE
        elif action >= _ACT_POWERS and action < _ACT_POWERS + self._game._num_power_cards:
            self._assign_power(action - _ACT_POWERS)
            self._after_power_choice() #find next player to pick power card, or move on one phase
        elif action >= _ACT_RETRIEVE_POWERS and action < _ACT_RETRIEVE_POWERS + self._game._num_power_cards:
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
        elif action >= _ACT_CHOOSE_SECRETS and action < _ACT_CHOOSE_SECRETS + self._game._num_cab_areas:
            selRegion = (action - _ACT_CHOOSE_SECRETS)
            self._set_secret_region(selRegion)
            if self._turn_state[_ST_TN_PHASE]==_ST_PHASE_SCORE:
                self._after_score_step()
            elif self._turn_state[_ST_TN_PHASE]==_ST_PHASE_RESPONSE:
                if self._rsp_steps[0]['do']=='toprovince':
                    #send one of your cabs to the province
                    self._quickmove_cabs(selRegion,self._game._province_idx,1)
                else:
                    #setting secret region - already done
                    pass
                #move on to next
                self._set_next_response_vars()
                self._after_action_step()
            else:
                #if we weren't chosing for cab movement in scoring, we were choosing for a card action
                self._apply_secret_choice(action)
                self._after_action_step() 
        elif action >= _ACT_MOVE_GRANDES and action < _ACT_MOVE_GRANDES + self._game._num_regions:
            self._move_grande(action - _ACT_MOVE_GRANDES) #1-step action always
            self._after_action_step() 
        elif action >= _ACT_MOVE_KINGS and action < _ACT_MOVE_KINGS + self._game._num_regions:
            self._move_king(action - _ACT_MOVE_KINGS) #1-step action always
            self._after_action_step() 
        elif action >= _ACT_MOVE_SCOREBOARDS and action < _ACT_MOVE_SCOREBOARDS + (self._game._num_scoreboards*self._game._num_regions):
            board = (action - _ACT_MOVE_SCOREBOARDS)//self._game._num_regions
            region = (action - _ACT_MOVE_SCOREBOARDS)%self._game._num_regions
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
            fromRegion = (action- _ACT_CAB_MOVES)//(self._game._num_cab_areas * _MAX_PLAYERS)
            toRegion = ((action- _ACT_CAB_MOVES)%(self._game._num_cab_areas * _MAX_PLAYERS))//_MAX_PLAYERS
            ofPlayer = (action- _ACT_CAB_MOVES)%_MAX_PLAYERS
            self._move_one_cab(fromRegion, toRegion, ofPlayer)        
            self._after_action_step() 
    
    def action_to_string(self, arg0, arg1=None, withPlayer=True):
        """Action -> string. Args either (player, action) or (action)."""
        player = self.current_player() if arg1 is None else arg0
        action = arg0 if arg1 is None else arg1
        return self._action_output(action,player,withPlayer,json=False)

    def _action_output(self,action,player,withPlayer=False,json=False):
        actionString=""
        jsondata=[]
        if action>=_ACT_CARDS and action < _ACT_CARDS + self._game._num_action_cards:
            cardname = self._game._cardtrack[action-_ACT_CARDS]
            actionString = "Action "+cardname
            jsondata={"action":cardname,"deck":cardname[:5]}
        elif action >= _ACT_POWERS and action < _ACT_POWERS + self._game._num_power_cards:
            actionString = "Power "+str(action + 1 - _ACT_POWERS)
            jsondata={"power":str(action + 1 - _ACT_POWERS)}
        elif action >= _ACT_RETRIEVE_POWERS and action < _ACT_RETRIEVE_POWERS + self._game._num_power_cards:
            actionString = "Retrieve Power "+str(action + 1 - _ACT_RETRIEVE_POWERS)
            jsondata={"retpower":str(action + 1 - _ACT_RETRIEVE_POWERS)}
        elif action == _ACT_DECIDE_CAB:
            actionString = "{0} placement before card action".format(self._game._config_data['units']['caballero'])
            jsondata={"decision":"cab"}
        elif action == _ACT_DECIDE_ACT:
            actionString = "Card action before {0} placement".format(self._game._config_data['units']['caballero'])
            jsondata={"decision":"card"}
        elif action == _ACT_DECIDE_ACT_ALT:
            actionString = "Card action (2nd choice) before {0} placement".format(self._game._config_data['units']['caballero'])
            jsondata={"decision":"card"}
        elif action >= _ACT_CHOOSE_SECRETS and action < _ACT_CHOOSE_SECRETS + self._game._num_cab_areas:
            actionString = "Choose "+ self._game._regions[action - _ACT_CHOOSE_SECRETS]
            jsondata={"choose":self._game._regions[action - _ACT_CHOOSE_SECRETS]}
        elif action >= _ACT_MOVE_GRANDES and action < _ACT_MOVE_GRANDES + self._game._num_regions:
            actionString = "{0} to {1}".format(self._game._config_data['units']['grande'],self._game._regions[action - _ACT_MOVE_GRANDES])
            jsondata={"grande":self._game._regions[action - _ACT_MOVE_GRANDES]}
        elif action >= _ACT_MOVE_KINGS and action < _ACT_MOVE_KINGS + self._game._num_regions:
            actionString = "{0} to {1}".format(self._game._config_data['units']['king'],self._game._regions[action - _ACT_MOVE_KINGS])
            jsondata={"king":self._game._regions[action - _ACT_MOVE_KINGS]}
        elif action >= _ACT_MOVE_SCOREBOARDS and action < _ACT_MOVE_SCOREBOARDS + (self._game._num_scoreboards*self._game._num_regions):
            board = (action - _ACT_MOVE_SCOREBOARDS)//self._game._num_regions
            region = (action - _ACT_MOVE_SCOREBOARDS)%self._game._num_regions
            actionString = "Move scoreboard {0} to {1}".format(self._scoreboards[board]['points'],self._game._regions[region])   
            jsondata={"board":self._scoreboards[board]['points'],"region":self._game._regions[region]}
        elif action == _ACT_SKIP:
            actionString = "Skip this step"
            jsondata={"decision":"skip"}
        elif action == _ACT_TRIGGER:
            actionString = "Trigger card action"
            jsondata={"decision":"trigger"}
        else:
            #moving a caballero fromregion, toregion, ofplayer
            fromRegion = (action- _ACT_CAB_MOVES)//(self._game._num_cab_areas * _MAX_PLAYERS)
            toRegion = ((action- _ACT_CAB_MOVES)%(self._game._num_cab_areas * _MAX_PLAYERS))//_MAX_PLAYERS
            ofPlayer = (action- _ACT_CAB_MOVES)%_MAX_PLAYERS
            actionString = "{0} {1} from {2} to {3}".format(self._players[ofPlayer],self._game._config_data['units']['caballero'],self._game._regions[fromRegion],self._game._regions[toRegion] )       
            jsondata={"cabfrom":self._game._regions[fromRegion],"cabto":self._game._regions[toRegion],"ofplayer":self._players[ofPlayer]}
 

        if json:
            jsondata["player"]=self._players[player]
            return jsondata
        elif withPlayer: 
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
        for i in range(self._game._num_cab_areas):
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
        my_copy._points = self._points.copy()
        my_copy._rsp_player = self._rsp_player
        my_copy._rsp_phase = self._rsp_phase
        my_copy._rsp_steps = self._rsp_steps.copy()
        my_copy._rsp_finalize = self._rsp_finalize
        my_copy._scoreboards = self._scoreboards.copy()
        my_copy._turn_state = self._turn_state.copy()
        my_copy._win_points = self._win_points.copy()
        my_copy._state_returns = self._state_returns.copy()
        my_copy._winner=self._winner
        return my_copy

_games=[]

class ElGrandeGame(pyspiel.Game):
    """El Grande Game
    """

    def __init__(self,params={"players":_DEFAULT_PLAYERS,"game_state":'',"game_state_json":'',"config_file":_DEFAULT_CONFIG}):
        super().__init__(_GAME_TYPE, _GAME_INFO, params)
        _games.append(self)

        self._config_file = _DEFAULT_CONFIG
        self._num_players=4


        if params.get("config_file",None) is not None:
            self._config_file=params["config_file"]
        
        if params.get("players",None) is not None:
            self._num_players=params["players"]
        game_state=''
        game_state_json=''
        
        if params.get("game_state",None) is not None:
            game_state=params["game_state"]
        if params.get("game_state_json",None) is not None:
            game_state_json=params["game_state_json"]

        #there is no need for _state and _state_json to both be given as parameters - if they are, use _state_json.
        #config_file is read from game state by preference, else from parameter, else use default
        if game_state == '' and game_state_json == '':
            self._game_state=''
        elif game_state_json != '':
            #parameter is actually a string - convert to json doc for compatibility
            self._game_state = json.loads(game_state_json)
            self._config_file = self._game_state.get('config',_DEFAULT_CONFIG)
        else:
            couch = couchdb.Server('http://'+self._couchcred+'@'+self._couchip)
            gamehistdb = couch['game_history']
            self._game_state = gamehistdb[game_state]
            self._config_file = self._game_state.get('config',_DEFAULT_CONFIG)
            
        self._init_game_config()

    def _init_game_config(self):
        #read in configuration information from file
        cfilename=self._config_file + ".json"
        with open(cfilename) as f:
            self._config_data=json.load(f)
        #couch defaults to localhost, or is settable from config
        self._couchip = self._config_data.get('couchip','127.0.0.1:5984')
        self._couchcred = self._config_data.get('couchcred','user:pass') 
        #express game info simply for ease of use calculating game states
        self._num_regions = len(self._config_data['regions'])
        self._castillo_idx = self._num_regions
        self._court_idx = self._num_regions+1
        self._province_idx = self._num_regions+2
        self._regions=self._config_data['regions']+[self._config_data['castillo'],self._config_data['court'],self._config_data['province']]
        self._region_ids = {self._config_data['regions'][i]:i for i in range(self._num_regions)}
        self._region_ids[self._config_data['castillo']]=self._castillo_idx
        self._region_ids[self._config_data['court']]=self._court_idx
        self._region_ids[self._config_data['province']]=self._province_idx

        self._num_ext_regions = self._num_regions+1
        self._num_cab_areas = self._num_ext_regions+2
        self._decks = self._config_data['decks']
        self._num_decks = len(self._config_data['decks'])
        self._deck_names = self._config_data['decks'].keys()
        self._cardtrack = [card for deck in self._decks for card in self._decks[deck]]
        self._num_action_cards = len(self._cardtrack)
        self._cards = self._config_data['cards']
        self._neighbors = {self._region_ids[r]:[self._region_ids[n] for n in self._config_data['neighbors'][r]] for r in self._config_data['regions']}
        self._points = {self._region_ids[r]:self._config_data['points'][r] for r in self._config_data['points']}
        self._scoreboards = [{'name':s,'points':self._config_data['scoreboards'][s]} for s in self._config_data['scoreboards']]
        self._power_cabs=[6,5,5,4,4,3,3,2,2,1,1,0,0] #won't allow this to vary at the moment
        self._num_power_cards = len(self._power_cabs)
        self._num_scoreboards = len(self._scoreboards)

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
        board_size=(_ST_IDCH*_MAX_CAB_AREAS*_ST_BDY_END)
        gk_size=(_MAX_REGIONS*(_MAX_PLAYERS+1))
        cards_size=(_MAX_ACTION_CARDS*2)
        phase_size=3

        self.tensor = np.zeros(board_size + gk_size + cards_size + phase_size, np.float32)
        self._board = self.tensor[:board_size].reshape(_ST_IDCH, _MAX_CAB_AREAS, _ST_BDY_END)
        self._gk = self.tensor[board_size:(board_size+gk_size)].reshape(_MAX_REGIONS,_MAX_PLAYERS+1)
        self._cards = self.tensor[(board_size+gk_size):(board_size+gk_size+cards_size)]
        self._phase = self.tensor[(board_size+gk_size+cards_size):]
        self.dict = {"board": self._board,"grande_king":self._gk,"cards":self._cards,"phase":self._phase}



    def set_from(self, state, player):
        del player
        for channel in range(_ST_IDCH):
            chmat = (state._board_state[:state._game._num_cab_areas,:_ST_BDY_END] >> channel)%2
            self._board[channel,:state._game._num_cab_areas,:_ST_BDY_END]=chmat

        for channel in range(_MAX_PLAYERS+1):
            chvec = (state._board_state[:state._game._num_cab_areas,_ST_BDY_GRANDE_KING] >> channel)%2
            self._gk[:,channel]=chvec

        cards_ud_cd = [state._acard_state[s] in [_ST_AC_CHOSEN,_ST_AC_DONE] for s in range(state._game._num_action_cards)]
        cards_uc_dd = [state._acard_state[s] in [_ST_AC_DEALT,_ST_AC_DONE] for s in range(state._game._num_action_cards)]
        self._cards[:state._game._num_action_cards]=cards_ud_cd
        self._cards[-state._game._num_action_cards:]=cards_uc_dd

        self._phase[:]=self._PHASES[state._turn_state[_ST_TN_PHASE]]

    def string_from(self, state, player):
        del player
        return str(state)

pyspiel.register_game(_GAME_TYPE,ElGrandeGame)
