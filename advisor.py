import base64
import logging
import os
import re
import requests
import sys
import couchdb
import json
import copy

couchip = '127.0.0.1:5984'
credentials = 'admin:elderberry'
couch = couchdb.Server('http://'+credentials+'@'+couchip)
#list of advice actions which should be generated for each relevant phase
adviceTypes = {'power':['opponent_report','score_predictions','explain_cards','home_report'], 
  'action':['caballero_value','card_action_value','opponent_report','score_predictions','explain_cards','home_report'],
  'scoring':['castillo','opponent_report','score_predictions']}
#regions, as in TTS code
regions=["Galicia","Pais Vasco","Aragon","Cataluna","Castilla la Vieja","Castilla la Nueva","Valencia","Sevilla","Granada"]
extRegions=regions.copy()
extRegions.append("Castillo")

#invoked when the Flask listener detects data coming in to the db
#assume that the data is 'body' text about to be forwarded to db - we inspect this to see what sort of
#advice needs to be generated. Input data should be a byte string with JSON
def processDBInput(inputData):
  assert(type(inputData)==bytes)
  jsonObject=json.loads(inputData.decode('utf-8'))
  if jsonObject.get('turninfo')!=None:
    generateAdviceFor(jsonObject)
    
#input data for a turn and phase - generate some advice, depending on what turn, phase and player we have here
#structure of jsonObject should be game_history
def generateAdviceFor(jsonObject):
  assert(jsonObject['turninfo']!=None)
  if len(jsonObject['turninfo']['playersleft'])>0:
    player=jsonObject['turninfo']['playersleft'][0]
    phase=jsonObject['turninfo']['phase']
    relevantAdvice = adviceTypes.get(phase,[])
    for atype in relevantAdvice:
      makeAdvice(player,atype,jsonObject)
      
 #generate specified advice for specified player/round/phase based on this jsonObject
def makeAdvice(player,atype,jsonObject):
    if atype=='caballero_value':
      advice = caballeroAdvice(player,jsonObject)
    elif atype=='card_action_value':
      advice = cardActionValueAdvice(player,jsonObject)
    elif atype=='castillo':
      advice = castilloAdvice(player,jsonObject)
    elif atype=='explain_cards':
      advice = explainCardsAdvice(player,jsonObject)
    elif atype=='home_report':
      advice = homeReportAdvice(player,jsonObject)
    elif atype=='opponent_report':
      advice = opponentReportAdvice(player,jsonObject)
    elif atype=='score_predictions':
      advice = scorePredictionsAdvice(player,jsonObject)
      
    try:
      adviceDB = couch['advice']
    except:
      adviceDB = couch.create('advice')
      
    adviceDB.save(advice)

def initAdviceStructure(player,jsonObject):
    return {'game':jsonObject['name'],'player':player,'round':jsonObject['turninfo']['round'],
                  'phase':jsonObject['turninfo']['phase'],'advicetype':'','advice':{}}

#value of each caballero in each region
#value of each caballero in each region
def caballeroAdvice(player,jsonObject):
    advice = initAdviceStructure(player,jsonObject)
    advice['advicetype']='caballero'
    for region in extRegions:
        advice['advice'][region]=assessCaballeroPoints(player,region,jsonObject)
    return advice
 
#possible point value of playing this card in this round
def cardActionValueAdvice(player,jsonObject):
    advice = initAdviceStructure(player,jsonObject)
    return advice
 
#consequences of putting castillo caballeros in each region
def castilloAdvice(player,jsonObject):
    advice = initAdviceStructure(player,jsonObject)
    advice['advicetype']='castillo'
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
    advice = initAdviceStructure(player,jsonObject)
    return advice
 
#whether player's home region is 'under threat' or they have already lost number 1 status there
def homeReportAdvice(player,jsonObject):
    advice = initAdviceStructure(player,jsonObject)
    return advice
 
#regions where each player has been scoring historically, whether it's first/second/third, and how many areas the player is scoring.
def opponentReportAdvice(player,jsonObject):
    advice = initAdviceStructure(player,jsonObject)
    return advice
 
#predicted scores for each player based on this round
def scorePredictionsAdvice(player,jsonObject):
    advice = initAdviceStructure(player,jsonObject)
    return advice
  
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
