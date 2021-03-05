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

couchip = '127.0.0.1:5984'
credentials = 'admin:elderberry'
couch = couchdb.Server('http://'+credentials+'@'+couchip)
#list of advice actions which should be generated for each relevant phase
adviceTypes = {'power':['opponent_report','score_predictions','explain_cards','home_report'], 
  'action':['caballero_value','card_action_value','opponent_report','score_predictions','explain_cards','home_report','suggestion'],
  'scoring':['castillo','opponent_report','score_predictions']}
#regions, as in TTS code
regions=["Galicia","Pais Vasco","Aragon","Cataluna","Castilla la Vieja","Castilla la Nueva","Valencia","Sevilla","Granada"]
extRegions=regions.copy()
extRegions.append("Castillo")

_ROUND_COUNT=10

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
#structure of jsonObject should be game_history
def generateAdviceFor(jsonObject, gameHistory):
  print('advice')
  assert(jsonObject.get('turninfo','')!='')
  thisGame = el_grande.ElGrandeGame({"game_state_json":pySpiel.GameParameter(json.dumps(jsonObject))})
  thisGameState = thisGame.new_initial_state()
  
  if jsonObject['turninfo']['phase']=='start':
    addRelevantGameHistory(thisGameState, gameHistory,jsonObject['name'])
    print("Game History Length {0}".format(len(gameHistory)))

  if len(jsonObject['turninfo']['playersleft'])>0:
    player=jsonObject['turninfo']['playersleft'][0]
    phase=jsonObject['turninfo']['phase']
    relevantAdvice = adviceTypes.get(phase,[])
    for atype in relevantAdvice:
      print(atype)
      proc = Process(target=makeAdvice,args=(player, atype, gameState, jsonObject, gameHistory))
      proc.start()
      proc.join()
      
 #generate specified advice for specified player/round/phase based on this gameState or jsonObject
def makeAdvice(player, atype, gameState, jsonObject, gameHistory):
    if atype=='caballero_value':
      advice = caballeroAdvice(player, jsonObject)
    elif atype=='card_action_value':
      advice = cardActionValueAdvice(player, jsonObject)
    elif atype=='castillo':
      advice = castilloAdvice(player, jsonObject)
    elif atype=='explain_cards':
      advice = explainCardsAdvice(player, jsonObject)
    elif atype=='home_report':
      advice = homeReportAdvice(player, jsonObject, gameHistory)
    elif atype=='opponent_report':
      advice = opponentReportAdvice(player, jsonObject, gameHistory)
    elif atype=='score_predictions':
      advice = scorePredictionsAdvice(player, jsonObject)
    elif atype=='suggestion':
      advice = suggestionAdvice(player, jsonObject)
      
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
def cardActionValueAdvice(player,jsonObject):
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
    return advice
 
#whether player's home region is 'under threat' or they have already lost number 1 status there
def homeReportAdvice(player, jsonObject, gameHistory):
    advice = initAdviceStructure(player, 'home_report', jsonObject)
    home = jsonObject['pieces'][player]['grande']
    round = jsonObject['turninfo']['round']
    homecabs={}
    for player in gameHistory[jsonObject['name']]:
        homecabs[player] = gameHistory[jsonObject['name']][player].get(home,np.zeros(_ROUND_COUNT))
        homecabs[player][round]=jsonObject['pieces'][player].get(home,0)

    #TODO: check for control of region
    #check for recent activity from not-you
    #check if it's in anyone's interest to overtake you
    return advice
 
#regions where each player has been scoring historically, whether it's first/second/third, and how many areas the player is scoring.
def opponentReportAdvice(player, jsonObject, gameHistory):
    advice = initAdviceStructure(player, 'opponent_report', jsonObject)
    return advice
 
#predicted scores for each player based on this round
def scorePredictionsAdvice(player, jsonObject):
    advice = initAdviceStructure(player, 'score_prediction', jsonObject)
    return advice
 
#suggestion of what cards to play
def suggestionAdvice(player, jsonObject):
    advice = initAdviceStructure(player, 'suggestion', jsonObject)
    return advice

#insert information about caballero movements in a way that will be
#easy to interpret for the advisor
def addRelevantGameHistory(gameState, gameHistory, gameName):

    round = gameState._get_round()
    regions = pieces._NUM_REGIONS
    players = gameState._num_players
    
    localHistory = gameHistory.get(gameName,np.zeros((regions,players,_ROUND_COUNT))
    localHistory[:,:,round] = gameState._board_state[:regions,:players]
    gameHistory[gameName]=localHistory
    #hoping the info is correctly stored in shared memory 
    return

 
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
