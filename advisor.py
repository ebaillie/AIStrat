import base64
import logging
import os
import re
import requests
import sys
import couchdb
import json

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
    player=jsonObject['turninfo']['playersleft'][1]
    round=jsonObject['turninfo']['round']
    phase=jsonObject['turninfo']['phase']
    relevantAdvice = adviceTypes.get(phase,[])
    for atype in relevantAdvice:
      makeAdvice(player,round,phase,atype,jsonObject)
      
 #generate specified advice for specified player/round/phase based on this jsonObject
 def makeAdvice(player,round,phase,atype,jsonObject):
    if atype=='caballero_value':
      advice = caballeroAdvice(player,round,phase,jsonObject)
    elif atype=='card_action_value':
      advice = cardActionValueAdvice(player,round,phase,jsonObject)
    elif atype=='castillo':
      advice = castilloAdvice(player,round,phase,jsonObject)
    elif atype=='explain_cards':
      advice = explainCardsAdvice(player,round,phase,jsonObject)
    elif atype=='home_report':
      advice = homeReportAdvice(player,round,phase,jsonObject)
    elif atype=='opponent_report':
      advice = opponentReportAdvice(player,round,phase,jsonObject)
    elif atype=='score_predictions':
      advice = scorePredictionsAdvice(player,round,phase,jsonObject)
      
    try:
      adviceDB = couch['advice']
    except:
      adviceDB = couch.create('advice')
      
    adviceDB.save(advice)


#value of each caballero in each region
def caballeroAdvice(player,round,phase,jsonObject):
 
#possible point value of playing this card in this round
def cardActionValueAdvice(player,round,phase,jsonObject):
 
#consequences of putting castillo caballeros in each region
def castilloAdvice(player,round,phase,jsonObject):
 
#human-generated explanations of use of current cards
def explainCardsAdvice(player,round,phase,jsonObject):
 
#whether player's home region is 'under threat' or they have already lost number 1 status there
def homeReportAdvice(player,round,phase,jsonObject):
 
#regions where each player has been scoring historically, whether it's first/second/third, and how many areas the player is scoring.
def opponentReportAdvice(player,round,phase,jsonObject):
 
#predicted scores for each player based on this round
def scorePredictionsAdvice(player,round,phase,jsonObject):
  
#generate all region ranking data for one region
def rankPlayersInRegion(regionName,jsonObject):
    ranks = {}
    pieces = {}
    excess = {}
    for k in jsonObject['pieces']:
        if regionName in jsonObject['pieces'][k]:
            pieces[k]=jsonObject['pieces'][k][regionName]

    thisCount=0
    pieceLevel=float('inf')
    lastKey=None
    for k,v in sorted(pieces.items(), key=lambda item: item[1],reverse=True):
        thisCount=thisCount+1
        currentVal=pieces.get(k,0)
        print(currentVal)
        
        #if the excess for the last key was not already specified (due to a draw), insert it now
        if lastKey and excess.get(lastKey,-1)==-1:
            excess[lastKey]=pieceLevel-currentVal
            
        if currentVal<pieceLevel:
            pieceLevel=currentVal
            thisRank=thisCount
        elif currentVal==pieceLevel:
            #excess for equal ranked players must be 1
            excess[lastKey]=1
            excess[k]=1
        else:
            #a terrible error has occurred - this should never be possible
            raise Exception("Player ranking algorithm has a bug - fix it!")
            
        if currentVal>0:
            ranks[k]=thisRank
        
        lastKey=k
    
    #make sure to enter excess data for the last player
    if lastKey and excess.get(lastKey,-1)==-1:
        excess[lastKey]=pieceLevel
        
    return ranks,pieces,excess
  
