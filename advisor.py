import base64
import logging
import os
import re
import requests
import sys
import couchdb
import json
import copy
import numpy as np
import pyspiel
import el_grande
import el_grande_pieces as pieces

from multiprocessing import Process, Manager

_ROUND_COUNT=10
_ADV_CURRENT = 0 #produce advice for current player
_ADV_EACH = 1 #produce separate advice for each player in game
_ADV_ALL = 2 #produce one advice accessible by all players

couchip = '127.0.0.1:5984'
credentials = 'admin:elderberry'
couch = couchdb.Server('http://'+credentials+'@'+couchip)
#list of advice actions which should be generated for each relevant phase
#stored as tuple: (type, phase, forPlayer)
adviceList = [('explain_cards','power',_ADV_ALL),
                ('opponent_report','start',_ADV_ALL),
                ('opponent_report','scoring',_ADV_ALL),
                ('home_report','start',_ADV_EACH),
                ('home_report','action',_ADV_CURRENT),
                ('caballero_value','action',_ADV_EACH),
                ('card_action_value','action',_ADV_CURRENT),
                ('score_predictions','action',_ADV_CURRENT),
                ('suggestion','action',_ADV_CURRENT),
                ('castillo','scoring',_ADV_EACH),
                ('alert','action',_ADV_EACH)]

#regions, as in TTS code
regions=["Galicia","Pais Vasco","Aragon","Cataluna","Castilla la Vieja","Castilla la Nueva","Valencia","Sevilla","Granada"]
extRegions=regions.copy()
extRegions.append("Castillo")


#invoked when the Flask listener detects data coming in to the db
#assume that the data is 'body' text about to be forwarded to db - we inspect this to see what sort of
#advice needs to be generated. Input data should be a byte string with JSON
def processDBInput(inputData, gameHistory):
  print('processing')
  assert(type(inputData)==bytes)
  jsonObject=json.loads(inputData.decode('utf-8'))
  if jsonObject.get('turninfo')!=None:
    generateAdviceFor(jsonObject, gameHistory)
    
#input data for a turn and phase - generate some advice, depending on what turn, phase and player we have here
def generateAdviceFor(jsonObject, gameHistory):
  print('advice')
  assert(jsonObject.get('turninfo','')!='')
  thisGame = el_grande.ElGrandeGame({"game_state_json":pyspiel.GameParameter(json.dumps(jsonObject))})
  thisGameState = thisGame.new_initial_state()
  
  if jsonObject['turninfo']['phase']=='start':
    addRelevantGameHistory(thisGameState, gameHistory,jsonObject['name'])
    print("Game History P0 pieces {0}".format(gameHistory[jsonObject['name']]['pieces'][:,0,:]))

  if len(jsonObject['turninfo']['playersleft'])>0:
    player=jsonObject['turninfo']['playersleft'][0]
    phase=jsonObject['turninfo']['phase']
    for adv in adviceList:
      if adv[1]==phase:
        print(adv[0])
        if adv[2]==_ADV_EACH:
            for p in jsonObject['players']:
                proc = Process(target=makeAdvice,args=(p, adv[0], thisGameState, jsonObject, gameHistory[jsonObject['name']]))
                proc.start()
                proc.join()
        else:
            p = player if adv[2]==_ADV_CURRENT else ''
            proc = Process(target=makeAdvice,args=(p, adv[0], thisGameState, jsonObject, gameHistory[jsonObject['name']]))
            proc.start()
            proc.join()
      
 #generate specified advice for specified player/round/phase based on this gameState or jsonObject
def makeAdvice(player, atype, gameState, jsonObject, thisHistory):
    if atype=='caballero_value':
      advice = caballeroAdvice(player, jsonObject)
    elif atype=='card_action_value':
      advice = cardActionValueAdvice(player, gameState, jsonObject)
    elif atype=='castillo':
      advice = castilloAdvice(player, jsonObject)
    elif atype=='explain_cards':
      advice = explainCardsAdvice(player, jsonObject)
    elif atype=='home_report':
      advice = homeReportAdvice(player, gameState, thisHistory, jsonObject)
    elif atype=='opponent_report':
      advice = opponentReportAdvice(player, gameState, thisHistory, jsonObject)
    elif atype=='score_predictions':
      advice = scorePredictionsAdvice(player, jsonObject)
    elif atype=='suggestion':
      advice = suggestionAdvice(player, jsonObject)
    elif atype=='alert':
      advice = alertAdvice(player, gameState, thisHistory, jsonObject)
      
    try:
      adviceDB = couch['game_advice']
    except:
      adviceDB = couch.create('game_advice')
      
    adviceDB.save(advice)

def initAdviceStructure(player, advicetype, jsonObject):
    return {'game':jsonObject['name'],'time':jsonObject['time'],'player':player,'round':jsonObject['turninfo']['round'],
                  'phase':jsonObject['turninfo']['phase'],'advicetype':advicetype,'advice':{}}

#value of each caballero in each region
def caballeroAdvice(player,jsonObject):
    advice = initAdviceStructure(player, 'caballero_value', jsonObject)
    for region in extRegions:
        advice['advice'][region]=assessCaballeroPoints(player,region,jsonObject)
    return advice
 
#possible point value of playing this card in this round
def cardActionValueAdvice(player,gameState, jsonObject):
    advice = initAdviceStructure(player, 'card_action_value', jsonObject)
    return advice
 
#consequences of putting castillo caballeros in each region
def castilloAdvice(player,jsonObject):
    advice = initAdviceStructure(player, 'castillo', jsonObject)
    pieces=copy.deepcopy(jsonObject['pieces'])
    castilloPieceCount=pieces[player].get('Castillo',0)
    if castilloPieceCount==0:
        return advice
    
    #counterfactuals for points available from putting castillo cabs in each region
    #simple strategy for the moment - only consider your own moves and points
    del pieces[player]['Castillo']
    for region in regions:
        curRanks,curPlayerPieces=rankPlayersInRegion(region,pieces)
        pieces[player][region]=pieces[player].get(region,0)+castilloPieceCount
        newRanks,newPlayerPieces=rankPlayersInRegion(region,pieces)
        if curRanks.get(player,4)>3:
            newScore=rankScore(newRanks[player],jsonObject['points'][region],pieces.get('grande','')==region,jsonObject['king']==region)
            advice['advice'][region]=newScore            
        elif curRanks.get(player,4)==newRanks[player]:
            advice['advice'][region]=0
        else:
            curScore=rankScore(curRanks[player],jsonObject['points'][region],pieces.get('grande','')==region,jsonObject['king']==region)
            newScore=rankScore(newRanks[player],jsonObject['points'][region],pieces.get('grande','')==region,jsonObject['king']==region)
            advice['advice'][region]=newScore-curScore
            
    return advice
 
#human-generated explanations of use of current cards
def explainCardsAdvice(player,jsonObject):
    advice = initAdviceStructure(player, 'explain_cards', jsonObject)
    adv_text=""
    for deck in jsonObject['cards']:
        card_id = jsonObject['cards'][deck]
        card_title = str.split(pieces._CARDS[card_id]["text"],":")[0]
        exp_string = pieces._CARDS[card_id]["explain"]
        adv_text = adv_text+card_title+": "+exp_string+"\n"
    advice['advice']=adv_text
    return advice
 
#whether player's home region is 'under threat' or they have already lost number 1 status there
def homeReportAdvice(player, gameState, thisHistory, jsonObject):
    advice = initAdviceStructure(player, 'home_report', jsonObject)
    player_id = gameState._get_pid(player)
    home = gameState._grande_region(player_id)
    round = gameState._get_round()
    #check for control of region, right now
    advice['advice']['control'] = (gameState._rank_region(home).get(player_id,-1) == 1)
    
    #check for recent activity from not-you
    moves = thisHistory['pieces'][home,:,round]-thisHistory['pieces'][home,:,(round-1)]
    advice['advice']['moves'] = [gameState._get_player_name(i) for i in range(gameState._num_players) if moves[i]>0 and i!=player_id]

    #check if it's in anyone's interest to overtake you
    targeters=[]
    for p in range(gameState._num_players):
        if p!=player_id:
            target = reportTarget(gameState, p, home)
            if target>0:
                targeters=targeters+[p]
    advice['advice']['targeters']=[gameState._get_player_name(t) for t in targeters]
    return advice
 
#regions where each player has been scoring historically, whether it's first/second/third, and how many areas the player is scoring.
def opponentReportAdvice(player, gameState, thisHistory, jsonObject):
    advice = initAdviceStructure(player, 'opponent_report', jsonObject)
    regions = pieces._NUM_EXT_REGIONS
    for r in range(regions):
        advice['advice'][pieces._REGIONS[r]]={'pieces':{},'ranks':{}}
        for p in range(gameState._num_players):
            currentPieces=gameState._board_state[r,p]
            placements=np.full(gameState._get_round()+1,0)
            ranks=[]
            trackCabs=0
            for rd in range(1,gameState._get_round()+1):
                placed=thisHistory['pieces'][r,p,rd] - trackCabs
                if placed>0:
                    placed = min(placed,(currentPieces-trackCabs))
                    placements[rd]=placed
                    trackCabs += placed
                if el_grande._SCORING_ROUND[rd]:
                    ranks = ranks + [thisHistory['ranks'][r,p,rd]]
            pname=gameState._get_player_name(p)
            if currentPieces>0:
                advice['advice'][pieces._REGIONS[r]]['pieces'][pname]=placements.tolist()
            advice['advice'][pieces._REGIONS[r]]['ranks'][pname]=ranks

    return advice
 
#predicted scores for each player based on this round
def scorePredictionsAdvice(player, jsonObject):
    advice = initAdviceStructure(player, 'score_prediction', jsonObject)
    return advice
 
#suggestion of what cards to play
def suggestionAdvice(player, jsonObject):
    advice = initAdviceStructure(player, 'suggestion', jsonObject)
    return advice

#specific alert of problems that might come up
def alertAdvice(player, gameState, thisHistory, jsonObject):
    advice = initAdviceStructure(player, 'alert', jsonObject)
    #home region lost this turn?
    player_id = gameState._get_pid(player)
    home = gameState._grande_region(player_id)
    round = gameState._get_round()
    #check for control of region, right now
    hasControl = (gameState._rank_region(home).get(player_id,-1) == 1)
    if not hasControl:
        #was control just lost?
        lastControl = thisHistory["ranks"][home,player_id,(round-1)] == 1
        if lastControl:
            advice['advice']['alert']='Lost control of home region' 
    return advice

#insert information about caballero movements in a way that will be
#easy to interpret for the advisor
def addRelevantGameHistory(gameState, gameHistory, gameName):

    round = gameState._get_round()
    regions = pieces._NUM_EXT_REGIONS
    players = gameState._num_players
   
    localHistory = gameHistory.get(gameName,{"pieces":np.zeros((regions,players,_ROUND_COUNT)),"ranks":np.zeros((regions,players,_ROUND_COUNT))}) 
    localHistory["pieces"][:,:,round] = gameState._board_state[:regions,:players]
    for r in range(regions):
        #rank everything last by default
        rd=gameState._rank_region(r)
        ranks=[rd.get(i,players) for i in range(players)]
        localHistory["ranks"][r,:,round]=ranks

    gameHistory[gameName]=localHistory
    #hoping the info is correctly stored in shared memory 
    return

#return in what order this region should be targeted by this player, using greedy algorithm
#don't worry about allowable placement - this is for goal setting
def reportTarget(gameState,player,region,stopAt=5):
    regions = pieces._NUM_EXT_REGIONS
    testPieces = np.full(regions,0) 
    for test in range(stopAt):
        #each round, work out how best to improve the point gap
        testState = gameState.clone()
        testState._board_state[:regions,player]+=testPieces
        defaultPoints=[testState._score_one_region(r) for r in range(regions)]
        defaultPointGap = getPointGap(sum(defaultPoints),player)
        bestPointGap = defaultPointGap #floor for improvement
        bestRegion = -1
        
        testState._board_state[:regions,player]+=1
        testPoints=[testState._score_one_region(r) for r in range(regions)]
        for r in range(regions):
            #check the effect on the point gap of subbing in each testPoints in turn
            testArray=[testPoints[reg] if reg==r else defaultPoints[reg] for reg in range(regions)]
            pointGap = getPointGap(sum(testArray),player)
            if pointGap>bestPointGap or (pointGap==bestPointGap and r==region):
                bestPointGap=pointGap
                bestRegion=r
        #once we identified best region for this round, stop if it's the one we want
        #otherwise register this in the array and keep going
        if bestRegion==region:
            return test
        else:
            testPieces[bestRegion]+=1

    #if we finished the loop without finding anything, return -1
    return -1

#if player is first, pointGap is +ve or 0 distance to next player
#if player is not first, pointGap is -ve distance to top player
def getPointGap(pointArray, player):
    top = max(pointArray)
    next = max([p for p in pointArray if p!=top])
    if pointArray[player]==top:
        #if equal firsts, gap is zero
        if len(np.where(pointArray==top)[0])>0:
            return 0
        else:
            return top-next
    else:
        return pointArray[player]-top
 
#generate all region ranking data for one region
#regionName - string
#pieces - dictionary of playername[regionName]:caballeroCount
def rankPlayersInRegion(regionName,pieces):
    ranks = {}
    playerPieces = {}

    for k in pieces:
        cPieces=pieces[k].get(regionName,0)
        if cPieces>0:
            playerPieces[k]=cPieces
            ranks[k]=1
            anchorRank=-1
            for k2 in playerPieces:
                if k!=k2:
                    #print("checking "+k+" against "+k2)
                    if playerPieces[k]>playerPieces[k2]:
                        #this new one is bigger, push the old one's rank down
                        ranks[k2]=ranks[k2]+1
                        #print("downranked "+k2)
                    elif playerPieces[k]==playerPieces[k2]:
                        #push the old one's rank down, and make a note of it for the end
                        ranks[k2]=ranks[k2]+1
                        anchorRank=ranks[k2]
                        #print("anchored to "+k2)
                    else:
                        #this one is smaller - downrank it
                        ranks[k]=ranks[k]+1
                        #print("downranked "+k)
                #we found an equal rank at some point - that becomes this one's final rank
                if anchorRank>0:
                    ranks[k]=anchorRank

    return ranks,playerPieces
  
#score region for rank and Grande/King constraints
#rank - numeric rank
#scores - tuple of (firstplace,secondplace,thirdplace)
def rankScore(rank,scores,isGrande=False,isKing=False):
    finalScore=0
    if rank>0 and rank<=3:
        finalScore=scores[rank-1] #fencepost
    
    if rank==1:
        if isGrande:
            finalScore=finalScore+2
        if isKing:
            finalScore=finalScore+2
    return finalScore

#determine value of having caballeros in this region, for player,
#conditional on the other players' caballero counts remaining unchanged
def assessCaballeroPoints(player,region,jsonObject):
    pieces=jsonObject['pieces']
    pointsPerPiece={}
    playerPieceCount=pieces[player].get(region,0)
    maxPlayerPieceCount=0
    for k in pieces:
        maxPlayerPieceCount=max(maxPlayerPieceCount,pieces[k].get(region,0))
    
    lastRank=4
    rankBound=0
    excess=0
    #calculate from one more than the current maximum, in order to capture all potential advantages
    #add one extra for fencepost reasons
    for i in list(range(1,maxPlayerPieceCount+2)):
        pieces[player][region]=i
        ranks,playerPieces=rankPlayersInRegion(region,pieces)
        #hold on to the point where the rank improves, if necessary
        if ranks[player]<lastRank:
            lastRank=ranks[player]
            rankBound=i
        if i==playerPieceCount:
            excess= i-rankBound
        #print("counterfactual ranking "+str(i))
        score=rankScore(ranks[player],jsonObject['points'][region],pieces.get('grande','')==region,jsonObject['king']==region)
        pointsPerPiece[i]=score/i
        
    #reset the counterfactual, because otherwise it messes with the initial data structure
    #avoiding having to do a deep copy
    pieceVal=0
    if playerPieceCount>0:
        pieces[player][region]=playerPieceCount
        pieceVal=pointsPerPiece[playerPieceCount]
    else:
        del pieces[player][region]
        
    
        
    maxval = max(pointsPerPiece.values())
    optimals = [k for k in pointsPerPiece if pointsPerPiece[k]==maxval]
    
    retObject={"pieces":playerPieceCount,"value":pieceVal,"excess":excess,"optimals":optimals,"optimalvalues":maxval}
    return retObject 
