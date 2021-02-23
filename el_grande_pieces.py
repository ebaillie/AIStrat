"""
Information about El Grande's initial board, power cards, action cards and movable scoreboards.
Most of this information remains unchanged for each game.
'_POINTS' and '_SCOREBOARDS' objects should be copied to game state rather than accessed directly, since they get some small modifications
whenever a movable scoreboard is moved to a region, to reflect the updated region score.

GUIDs, where present, are those allocated to the objects in the Tabletop Simulator version of the game. 
In the context of the Python version, they are simply arbitrary labels, but should remain unchanged for compatibility purposes

"""


_NUM_REGIONS = 9 #main board regions
_CASTILLO = 9 #castillo index
_COURT = 10 #court index
_PROVINCE = 11 #province index

_NUM_EXT_REGIONS = _CASTILLO #regions including castillo
_NUM_CAB_AREAS = _NUM_EXT_REGIONS + 2 #areas caballeros can be placed, including court and province
_NUM_FULL_DECKS = 4 #decks with multiple cards in them
_NUM_PLAYABLE_DECKS = 5 #include also the single-card 'Deck5'
_MAX_DECK_COUNT = 11 #greatest number of cards in any single deck
_NUM_ACTION_CARDS = 43 #total number of action cards in all decks
_NUM_POWER_CARDS = 13
_POWER_CABS = [6,5,5,4,4,3,3,2,2,1,1,0,0]
_DECK_ENDS = [11,9,11,11,1] #hard code number of cards per deck


_REGIONS = ['Aragon', 'Castilla la Nueva', 'Castilla la Vieja', 'Cataluna', 'Galicia', 'Granada', 'Pais Vasco', 'Sevilla', 'Valencia', 'Castillo', 'court', 'province']

_NEIGHBORS = [[1,2,3,6,8], [0,2,5,7,8], [0,1,4,6], [0, 8], [2,6], [1,7,8], [0,2,4], [1,5], [0,1,3,5]]

_POINTS = [[5, 4, 1], [7, 4, 2], [6, 4, 2], [4, 2, 1], [4, 2, 0], [6, 3, 1], [5, 3, 1], [4, 3, 1], [5, 3, 2], [5, 3, 1]]

_SCOREBOARDS = [{'guid': '1d084c', 'points': [4, 0, 0]}, {'guid': '32ebea', 'points': [8, 4, 0]}]

_DECKTRACK = {
                'Deck1':['18741e','216892','314091','35e53f','5fb89e','64eb17','6d9e0b','94545b','bf04b8','dd9df5','f0732a'],
                'Deck2':['15183c','47eac8','545212','59229c','752a2f','b2db04','b43391','bc81f0','c41142'],
                'Deck3':['0f9fd8','177822','1e4af7','2a46c4','2ce187','3776af','4c04df','6c5ce9','75a572','8a009f','a593e8'],
                'Deck4':['5612bf','689ccb','72a196','8c8c52','a4ec3f','a940a8','c9b4c3','d7eecb','dc5c52','e5dc76','f0c7ef'],
                'Deck5':['9ac840']
                }

_CARDTRACK = ['18741e','216892','314091','35e53f','5fb89e','64eb17','6d9e0b','94545b','bf04b8','dd9df5','f0732a','15183c','47eac8','545212','59229c','752a2f','b2db04','b43391','bc81f0','c41142','0f9fd8','177822','1e4af7','2a46c4','2ce187','3776af','4c04df','6c5ce9','75a572','8a009f','a593e8','5612bf','689ccb','72a196','8c8c52','a4ec3f','a940a8','c9b4c3','d7eecb','dc5c52','e5dc76','f0c7ef','9ac840']

_CARDS ={
            '18741e': {'idx':0 ,'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'all'}, 'number': 5, 'numopt': 'lteq', 'player': 'any', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}, 'name': 'Deck1_5Any_2', 'text': 'Intrigue: Choose a region. You may move 5 Caballeros from this region to other regions.'}, 
            '216892': {'idx':1, 'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'split'}, 'number': 4, 'numopt': 'lteq', 'player': 'any', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}, 'name': 'Deck1_4Any', 'text': 'Intrigue: You may move 4 Caballeros.'}, 
            '314091': {'idx':2, 'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'split'}, 'number': 4, 'numopt': 'lteq', 'player': 'self', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}, 'name': 'Deck1_4Self', 'text': 'Intrigue: You may move 4 of your Caballeros.'}, 
            '35e53f': {'idx':3, 'actiontype': 'all', 'details': [{'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'split'}, 'number': 2, 'player': 'self', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}}, {'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'split'}, 'number': 2, 'player': 'foreign', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}}], 'name': 'Deck1_2And2_1', 'text': 'Intrigue: You may move 2 of your Caballeros and 2 foreign Caballeros.'}, 
            '5fb89e': {'idx':4, 'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'split'}, 'number': 3, 'numopt': 'lteq', 'player': 'foreign', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}, 'name': 'Deck1_3Foreign', 'text': 'Intrigue: You may move 3 foreign Caballeros.'}, 
            '64eb17': {'idx':5, 'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'split'}, 'number': 3, 'numopt': 'lteq', 'player': 'any', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}, 'name': 'Deck1_3Any', 'text': 'Intrigue: You may move 3 Caballeros.'}, 
            '6d9e0b': {'idx':6, 'actiontype': 'move', 'details': {'from': {'region': 'court', 'splitopt': 'all'}, 'number': 2, 'numopt': 'lteq', 'player': 'self', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}, 'name': 'Deck1_Add', 'text': 'Intrigue: You may place 2 additional Caballeros from your Court into the region(s) of your choice.'}, 
            '94545b': {'idx':7, 'actiontype': 'all', 'details': [{'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'split'}, 'number': 2, 'player': 'self', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}}, {'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'split'}, 'number': 2, 'player': 'foreign', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}}], 'name': 'Deck1_2And2_2', 'text': 'Intrigue: You may move 2 of your Caballeros and 2 foreign Caballeros.'}, 
            'bf04b8': {'idx':8, 'actiontype': 'choose', 'details': [{'details': {'from': {'region': 'selfchoose', 'splitopt': 'all'}, 'number': -1, 'player': 'self', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}, 'type': 'move'}, {'details': {'from': {'region': 'court', 'splitopt': 'all'}, 'number': 2, 'player': 'self', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}, 'type': 'move'}], 'name': 'Deck1_Or', 'text': 'Intrigue: You may move all your Caballeros away from one region OR You may place 2 Caballeros from your Court into the region(s) of your choice.'}, 
            'dd9df5': {'idx':9, 'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'all'}, 'number': 5, 'numopt': 'lteq', 'player': 'any', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}, 'name': 'Deck1_5Any_1', 'text': 'Intrigue: Choose a region. You may move 5 Caballeros from this region to other regions.'}, 
            'f0732a': {'idx':10, 'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'all'}, 'number': -1, 'numopt': 'lteq', 'player': 'self', 'to': {'region': 'selfchoose', 'splitopt': 'split'}}, 'name': 'Deck1_Allself', 'text': 'Intrigue: You may move all your Caballeros away from one region.'}, 
            '15183c': {'idx':11, 'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'split'}, 'number': 1, 'numopt': 'lteq', 'player': 'foreign', 'to': {'region': 'province', 'splitopt': 'all'}}, 'name': 'Deck2_Provinceone', 'text': 'You may return 1 Caballero of each opponent to the Province. These Caballeros must come from regions.'}, 
            '47eac8': {'idx':12, 'actiontype': 'move', 'details': {'from': {'region': 'court', 'splitopt': 'all'}, 'number': 3, 'numopt': 'eq', 'player': 'foreign', 'to': {'region': 'province', 'splitopt': 'all'}}, 'name': 'Deck2_Decay', 'text': 'Decay of Authority: Each of your opponents must return 3 Caballeros in their Court to the Province.'}, 
            '545212': {'idx':13, 'actiontype': 'move', 'details': {'from': {'condition': 'clock', 'region': 'ownerchooseplus', 'splitopt': 'split'}, 'number': 3, 'numopt': 'each', 'player': 'foreign', 'to': {'region': 'province', 'splitopt': 'all'}}, 'name': 'Deck2_Angry', 'text': 'Angry King!: In clockwise order, each of your opponents must return 3 of their Caballeros to the Province. These Caballeros may come from their Court and/or a region on the board.'}, 
            '59229c': {'idx':14, 'actiontype': 'move', 'details': {'from': {'region': 'court', 'splitopt': 'all'}, 'number': -1, 'numopt': 'eq', 'player': 'foreign', 'to': {'region': 'province', 'splitopt': 'all'}}, 'name': 'Deck2_Decayall', 'text': 'Decay of Authority: Each of your opponents must return all Caballeros in their Court to the Province.'}, 
            '752a2f': {'idx':15, 'actiontype': 'score', 'details': {'number': 1, 'region': 'selfchoose'}, 'name': 'Deck2_Choose_2', 'text': 'Special Scoring: Choose a region to be scored immediately.'}, 
            'b2db04': {'idx':16, 'actiontype': 'move', 'details': {'from': {'condition': '2', 'region': 'ownerchoose', 'splitopt': 'all'}, 'number': 2, 'numopt': 'eq', 'player': 'any', 'to': {'region': 'province', 'splitopt': 'all'}}, 'name': 'Deck2_Province', 'text': 'Each player chooses a region (where they have at least 2 Caballeros) with their secret disc. Each player must return 2 of their Caballeros from their chosen region to the Province.'}, 
            'b43391': {'idx':17, 'actiontype': 'score', 'details': {'number': 1, 'region': 'selfchoose'}, 'name': 'Deck2_Choose_3', 'text': 'Special Scoring: Choose a region to be scored immediately.'}, 
            'bc81f0': {'idx':18, 'actiontype': 'move', 'details': {'from': {'condition': '1', 'region': 'ownerchoose', 'splitopt': 'all'}, 'number': -1, 'numopt': 'eq', 'player': 'any', 'to': {'region': 'province', 'splitopt': 'all'}}, 'name': 'Deck2_Provinceall', 'text': 'Each player chooses a region (where they have at least one Caballero) with their secret disc. Each player must return all their Caballeros from their chosen region to the Province.'}, 
            'c41142': {'idx':19, 'actiontype': 'score', 'details': {'number': 1, 'region': 'selfchoose'}, 'name': 'Deck2_Choose_1', 'text': 'Special Scoring: Choose a region to be scored immediately.'},
            '0f9fd8': {'idx':20, 'actiontype': 'score', 'details': {'number': -1, 'region': 'fives'}, 'name': 'Deck3_Fives_2', 'text': 'Special Scoring: All 5-point regions are scored immediately.'}, 
            '177822': {'idx':21, 'actiontype': 'score', 'details': {'number': -1, 'region': 'fours'}, 'name': 'Deck3_Fours_2', 'text': 'Special Scoring: All 4-point regions are scored immediately.'}, 
            '1e4af7': {'idx':22, 'actiontype': 'score', 'details': {'number': 1, 'region': 'selfchoose'}, 'name': 'Deck3_Choose', 'text': 'Special Scoring: Choose a region to be scored immediately.'}, 
            '2a46c4': {'idx':23, 'actiontype': 'score', 'details': {'number': -1, 'region': 'fours'}, 'name': 'Deck3_Fours_1', 'text': 'Special Scoring: All 4-point regions are scored immediately.'}, 
            '2ce187': {'idx':24, 'actiontype': 'score', 'details': {'number': -1, 'region': 'fives'}, 'name': 'Deck3_Fives_1', 'text': 'Special Scoring: All 5-point regions are scored immediately.'}, 
            '3776af': {'idx':25, 'actiontype': 'score', 'details': {'number': 1, 'region': 'castillo'}, 'name': 'Deck3_Castillo_2', 'text': 'Special Scoring: The Castillo is scored immediately.'},
            '4c04df': {'idx':26, 'actiontype': 'score', 'details': {'condition': 'top', 'number': -1}, 'name': 'Deck3_Firsts', 'text': 'Special Scoring: All regions are scored immediately. Only first place scores points. In case of a tie, no points are scored.'}, 
            '6c5ce9': {'idx':27, 'actiontype': 'score', 'details': {'number': -1, 'region': 'locab'}, 'name': 'Deck3_Least', 'text': 'Special Scoring: All regions with the least Caballeros are scored immediately. Regions containing no Caballeros do not count.'}, 
            '75a572': {'idx':28, 'actiontype': 'score', 'details': {'number': -1, 'region': 'sixsevens'}, 'name': 'Deck3_SixSevens', 'text': 'Special Scoring: All 6- and 7-point regions are scored immediately.'}, 
            '8a009f': {'idx':29, 'actiontype': 'score', 'details': {'number': -1, 'region': 'hicab'}, 'name': 'Deck3_Most', 'text': 'Special Scoring: All regions with the most Caballeros are scored immediately.'}, 
            'a593e8': {'idx':30, 'actiontype': 'score', 'details': {'number': 1, 'region': 'castillo'}, 'name': 'Deck3_Castillo_1', 'text': 'Special Scoring: The Castillo is scored immediately.'},
            '5612bf': {'idx':31, 'actiontype': 'scoreboard', 'name': 'Deck4_Scoreboard_2', 'text': 'Mobile Scoreboard: You may place a new mobile scoreboard OR move an existing one.'}, 
            '689ccb': {'idx':32, 'actiontype': 'power', 'name': 'Deck4_Power_2', 'text': 'Power of the Cards: You may return 1 of your used Power cards to your hand.'}, 
            '72a196': {'idx':33, 'actiontype': 'move', 'details': {'from': {'region': 'selfchoose', 'splitopt': 'all'}, 'number': -1, 'numopt': 'eq', 'player': 'foreign', 'to': {'region': 'ownerchoose', 'splitopt': 'all'}}, 'name': 'Deck4_Eviction', 'text': 'Eviction: Choose a region. Each of your opponents chooses a region on their secret disc. Each of your opponents moves all their Caballeros from this region to their chosen region.'}, 
            '8c8c52': {'idx':34, 'actiontype': 'scoreboard', 'name': 'Deck4_Scoreboard_3', 'text': 'Mobile Scoreboard: You may place a new mobile scoreboard OR move an existing one.'}, 
            'a4ec3f': {'idx':35, 'actiontype': 'move', 'details': {'from': {'region': 'province', 'splitopt': 'all'}, 'number': 2, 'numopt': 'lteq', 'player': 'self', 'to': {'region': 'court', 'splitopt': 'all'}}, 'name': 'Deck4_Court', 'text': 'The Court: You may take 2 Caballeros from the Province into your Court.'}, 
            'a940a8': {'idx':36, 'actiontype': 'grande', 'name': 'Deck4_Grande_2', 'text': 'Grande: You may move your Grande.'}, 
            'c9b4c3': {'idx':37, 'actiontype': 'scoreboard', 'name': 'Deck4_Scoreboard_1', 'text': 'Mobile Scoreboard: You may place a new mobile scoreboard OR move an existing one.'}, 
            'd7eecb': {'idx':38, 'actiontype': 'uniquescore', 'name': 'Deck4_Special', 'text': 'Special Scoring: Each player chooses a region with their secret disc. Each region chosen exactly once is immediately scored. Regions not chosen or chosen more than once are not scored.'}, 
            'dc5c52': {'idx':39, 'actiontype': 'king', 'details': {'condition': 'neighbor'}, 'name': 'Deck4_Advisor', 'text': 'Royal Advisor: You may move the King to an adjacent region.'}, 
            'e5dc76': {'idx':40, 'actiontype': 'grande', 'name': 'Deck4_Grande_1', 'text': 'Grande: You may move your Grande.'}, 
            'f0c7ef': {'idx':41, 'actiontype': 'power', 'name': 'Deck4_Power_1', 'text': 'Power of the Cards: You may return 1 of your used Power cards to your hand.'},
            '9ac840': {'idx':42, 'actiontype': 'king', 'details': {'condition': 'all'}, 'name': 'Deck5_King', 'text': "King's Card: You may move the King to any region."}} 

