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
import pickle

import pyspiel
import el_grande
import el_grande_pieces as pieces
import mcts_ext as mcts                        #local version of openspiel's mcts, with reporting extensions
import castillo_game

from multiprocessing import Process
from datetime import datetime

_LOG_FILE = "advisor.log"
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
                ('score_predictions','power',_ADV_EACH),
                ('suggestion','power',_ADV_CURRENT),
                ('opponent_report','start',_ADV_ALL),
                ('home_report','start',_ADV_EACH),
                ('home_report','action',_ADV_EACH),
                ('caballero_value','action',_ADV_EACH),
                ('card_action_value','action',_ADV_CURRENT),
                ('opponent_report','action',_ADV_ALL),
                ('score_predictions','action',_ADV_EACH),
                ('suggestion','action',_ADV_CURRENT),
                ('castillo','scoring',_ADV_EACH),
                ('opponent_report','scoring',_ADV_ALL),
                ('alerts','action',_ADV_EACH)]

#regions, as in TTS code
regions=["Galicia","Pais Vasco","Aragon","Cataluna","Castilla la Vieja","Castilla la Nueva","Valencia","Sevilla","Granada"]
extRegions=regions.copy()
extRegions.append("Castillo")

def makePickleName(jsonObject):
    return "pickles/"+jsonObject['name']+str(jsonObject['time'])

#invoked when the Flask listener detects data coming in to the db
#assume that the data is 'body' text about to be forwarded to db - we inspect this to see what sort of
#advice needs to be generated. Input data should be a byte string with JSON
def processDBInput(inputData):

  log('processing')
  assert(type(inputData)==bytes)
  jsonObject=json.loads(inputData.decode('utf-8'))
  
  #standard advice each turn
  if jsonObject.get('turninfo')!=None:
    if jsonObject['turninfo']['phase']=='end':
      archiveAdvice(jsonObject)
    else:
      generateAdviceFor(jsonObject)

  #specially requested advice
  if jsonObject.get('requestdata')!=None:
    generateRequestedAdvice(jsonObject)


   
#send all current advice to an archive table, then delete and re-create game_advice
#yes this means two concurrent games would have old advice archived before the end of game - not necessarily a problem
def archiveAdvice(jsonObject):

    try:
      adviceDB = couch['game_advice']
    except:
      #no advice to archive - exit
      return
      
    try:
      advicearchiveDB = couch['game_advice_archive']
    except:
      advicearchiveDB = couch.create('game_advice_archive')
      
    for advice_id in adviceDB:
      advice = adviceDB[advice_id]
      del advice['_id']
      del advice['_rev']
      advicearchiveDB.save(advice)

    couch.delete('game_advice')
    couch.create('game_advice')
 
#input data for a turn and phase - generate some advice, depending on what turn, phase and player we have here
def generateAdviceFor(jsonObject):
  log('advice')
  assert(jsonObject.get('turninfo','')!='')
  thisGame = el_grande.ElGrandeGame({"game_state_json":pyspiel.GameParameter(json.dumps(jsonObject))})
  thisGameState = thisGame.new_initial_state()
  players = thisGameState._num_players
  regions = pieces._NUM_EXT_REGIONS
 
  gameHistory = {"pieces":np.zeros((regions,players,_ROUND_COUNT)),"ranks":np.zeros((regions,players,_ROUND_COUNT))}
  pickleName = makePickleName(jsonObject)
  try:
    gameHistory = pickle.load( open( pickleName, "rb" ) )
    log("unpickled game history")
  except:
    log("blank game history")

  if jsonObject['turninfo']['phase']=='start':
    addRelevantGameHistory(thisGameState, gameHistory)
    pickle.dump( gameHistory, open( pickleName, "wb" ) )
    log("Game History P0 pieces {0}".format(gameHistory['pieces'][:,0,:]))

  if len(jsonObject['turninfo']['playersleft'])>0:
    player=jsonObject['turninfo']['playersleft'][0]
    phase=jsonObject['turninfo']['phase']
    for adv in adviceList:
      if adv[1]==phase:
        log(adv[0])
        if adv[2]==_ADV_EACH:
            for p in jsonObject['players']:
                makeAdvice(p, adv[0], thisGameState, jsonObject, gameHistory)
        else:
            p = player if adv[2]==_ADV_CURRENT else ''
            makeAdvice(p, adv[0], thisGameState, jsonObject, gameHistory)
      
#generate specified advice for specified player/round/phase based on this gameState or jsonObject
def makeAdvice(player, atype, gameState, jsonObject, thisHistory):
    if player!='' and jsonObject['advisor'][player]['on']=='false':
       return 

    proc = None
    advice = initAdviceStructure(player, atype, jsonObject)
    if atype=='caballero_value':
      proc = Process(target=caballeroAdvice,args=(player, gameState, jsonObject, advice))
    elif atype=='card_action_value':
      proc = Process(target=cardActionValueAdvice,args=(player, gameState, jsonObject, advice))
    elif atype=='castillo':
      proc = Process(target=castilloAdvice,args=(player, gameState, jsonObject, advice))
    elif atype=='explain_cards':
      proc = Process(target=explainCardsAdvice,args=(player, jsonObject, advice))
    elif atype=='home_report':
      proc = Process(target=homeReportAdvice,args=(player, gameState, thisHistory, jsonObject, advice))
    elif atype=='opponent_report':
      proc = Process(target=opponentReportAdvice,args=(player, gameState, thisHistory, jsonObject, advice))
    elif atype=='score_predictions':
      proc = Process(target=scorePredictionsAdvice,args=(player, gameState, jsonObject, advice))
    elif atype=='suggestion':
      proc = Process(target=suggestionAdvice,args=(player, gameState, jsonObject, advice))
    elif atype=='alerts':
      proc = Process(target=alertAdvice,args=(player, gameState, thisHistory, jsonObject, advice))
    proc.start()
    proc.join()

 
def saveAdvice(advice): 
    try:
      adviceDB = couch['game_advice']
    except:
      adviceDB = couch.create('game_advice')
    
    if len(advice['advice'])>0:  
        adviceDB.save(advice)

#special user-generated request for advice
def generateRequestedAdvice(requestJSON):
    #find info for the current round, which should already be in the database
    dbquery={"selector": {"name": {"$eq": requestJSON["name"]},"time": {"$eq": requestJSON["time"]},
            "turninfo": {"phase": {"$eq": requestJSON["phase"]},"round": {"$eq": requestJSON["round"]}}}}

    histDB = couch['game_history']
    hist_docs = histDB.find(dbquery)
    foundDoc = None 
    for hist in hist_docs:
        if requestJSON.get("player","")=="":
            #not filtered by player - by default this is an acceptable doc
            foundDoc = hist
            break
        elif len(hist["turninfo"]["playersleft"])>0 and hist["turninfo"]["playersleft"][0]==requestJSON["player"]:
            #filtered by player - make sure they're the next on turn
            foundDoc = hist
            break
    
    advice = {'game':requestJSON['name'],'time':requestJSON['time'],'player':requestJSON['player'],
            'round':requestJSON['round'],'phase':requestJSON['phase'],'trigger':'request',
            'advicetype':requestJSON['advicetype'],'requestdata':requestJSON['requestdata'],'advice':{}}
    if foundDoc == None:
        advice['advice'] = ["Problem finding game history data for " + requestJSON['advicetype'] + " advice."]
    else:
        thisGame =  el_grande.ElGrandeGame({"game_state_json":pyspiel.GameParameter(json.dumps(foundDoc))})
        thisState = thisGame.new_initial_state()
        if requestJSON['advicetype']=='suggestion':
            #find which of the legal actions matches the requested trial action, and play it.
            playedAct = False
            for act in thisState.legal_actions():
                if thisState.action_to_string(act,withPlayer=False)==requestJSON['requestdata']:
                    thisState.do_apply_action(act)
                    playedAct = True
                    break
            if playedAct:
                suggestion = initAdviceStructure(requestJSON['player'], 'suggestion', foundDoc)
                suggestionAdvice(requestJSON['player'],thisState,foundDoc,suggestion,False)
                advice['advice'] = [requestJSON['requestdata']] + suggestion['advice']
            else:
                advice['advice'] = ["Unable to simulate suggested play "+requestJSON['requestdata']] 

    saveAdvice(advice)
    


def initAdviceStructure(player, advicetype, jsonObject):
    triggerPlayer=""
    if len(jsonObject['turninfo']['playersleft'])>0:
        triggerPlayer=jsonObject['turninfo']['playersleft'][0]

    return {'game':jsonObject['name'],'time':jsonObject['time'],'player':player,'round':jsonObject['turninfo']['round'],
                  'phase':jsonObject['turninfo']['phase'],'trigger':triggerPlayer,'advicetype':advicetype,'advice':{}}

#value of each caballero in each region
def caballeroAdvice(player, gameState, jsonObject, advice):
    advice['advice']['regions']={}
    for region in extRegions:
        advice['advice']['regions'][region]=assessCaballeroPoints(player,region,jsonObject)

    #best places to put your cabs next:
    validTargets = pieces._NEIGHBORS[gameState._king_region()]+[pieces._CASTILLO]
    targets = reportTargets(gameState, gameState._get_pid(player), validList=validTargets)
    advice['advice']['targets']=[pieces._REGIONS[r] for r in targets if r>=0]
    saveAdvice(advice)
 
#possible point value of playing specific cards in this round
def cardActionValueAdvice(player, gameState, jsonObject, advice):
    
    mc_sims= 10
    mc_runs= 5
    turn = gameState._get_round()
    phase = gameState._get_current_phase_name()
    simGame = gameState.clone()
    simGame._end_turn = 3*((turn-1)//3) + 3 #next scoring round from where we are
    rng = np.random.RandomState()
    eval=mcts.RandomRolloutEvaluator(4,rng)
 
    #if we're before action phase for this player, skip, simulating getting first choice if possible
    while not (simGame._get_current_phase_name()=='action' and simGame.current_player()==simGame._get_pid(player)):
        act=min(simGame.legal_actions())
        if simGame.current_player()==simGame._get_pid(player): #power phase for current player
            act = max(simGame.legal_actions())
        simGame.do_apply_action(act)

    #get allowed actions
    legalActions = simGame.legal_actions()
    for action in legalActions:
        actionName = simGame.action_to_string(action,withPlayer=False)
        actVals=[]
        mbot = mcts.MCTSBot(simGame._game,2,mc_sims,eval,random_state=rng,solve=True,verbose=False)
        for run in range(mc_runs):
            testGame = simGame.clone()
            testGame.do_apply_action(action)
            act, actList, returns = mbot.multi_step(testGame)
            actVals = actVals+[returns[-1]]
        actMean=np.mean(actVals)
        actSD=np.std(actVals)
        advice['advice'][actionName]={"lo":round(actMean-2*actSD,1),"hi":round(actMean+2*actSD,1)}
    
    saveAdvice(advice)
 
#consequences of putting castillo caballeros in each region
def castilloAdvice(player, gameState, jsonObject, advice):
    cpieces=copy.deepcopy(jsonObject['pieces'])
    castilloPieceCount=cpieces[player].get('Castillo',0)
    if castilloPieceCount==0:
        return 
    
    #openSpiel Monte Carlo simulations
    playerID = gameState._get_pid(player)
    cabMovers = [p for p in range(gameState._num_players) if gameState._board_state[pieces._CASTILLO,p]>0]
    if len(cabMovers)>1:
        #more than one player moving cabs - it's worth simulating
        advice['advice']['simulations']={}
        sim_count=5
        choices=np.full((gameState._num_players,gameState._num_players*sim_count),0)
        for sim_player in range(gameState._num_players):
            temp_choices = np.full((gameState._num_players,sim_count),0)
            cg=castillo_game.CastilloGame({"state":pyspiel.GameParameter(gameState.castillo_game_string(sim_player))})
            cgs=cg.new_initial_state()
            rng=np.random.RandomState()
            eval=mcts.RandomRolloutEvaluator(1,rng)
            mbot = mcts.MCTSBot(cg,2,500,eval,random_state=rng,solve=True,verbose=False)
            for i in range(sim_count):
                c,cc,vc=mbot.multi_step(cgs,True) #do sim runs, reporting back whole game history
                optpad=np.full(gameState._num_players-len(cc),9).tolist() #pad out truncated multi-steps, where not all players have castillo pieces
                temp_choices[:,i]=cc+optpad
            #convert back to indexing by core game player numbers
            idxs=gameState.scoring_order(sim_player)
            for p in idxs:
                choices[p,sim_player*sim_count:(sim_player+1)*sim_count]=temp_choices[idxs[p]]

        #now convert from gameState to TTS format
        #for the moment only send data for self
        unique, counts = np.unique(choices[playerID,:], return_counts=True)
        pcount = dict(zip(unique.tolist(), (counts/sum(counts)).tolist()))
        advice['advice']['simulations'][gameState._players[playerID]]={pieces._REGIONS[p]:pcount[p] for p in pcount}

    #counterfactuals for points available from putting castillo cabs in each region
    #simple strategy - only consider your own moves and points
    
    advice['advice']['simple']={}
    del cpieces[player]['Castillo']
    for region in regions:
        curRanks,curPlayerPieces=rankPlayersInRegion(region,cpieces)
        cpieces[player][region]=cpieces[player].get(region,0)+castilloPieceCount
        newRanks,newPlayerPieces=rankPlayersInRegion(region,cpieces)
        if curRanks.get(player,4)>3:
            newScore=rankScore(newRanks[player],jsonObject['points'][region],cpieces.get('grande','')==region,jsonObject['king']==region)
            advice['advice']['simple'][region]=newScore            
        elif curRanks.get(player,4)==newRanks[player]:
            advice['advice']['simple'][region]=0
        else:
            curScore=rankScore(curRanks[player],jsonObject['points'][region],cpieces.get('grande','')==region,jsonObject['king']==region)
            newScore=rankScore(newRanks[player],jsonObject['points'][region],cpieces.get('grande','')==region,jsonObject['king']==region)
            advice['advice']['simple'][region]=newScore-curScore
            
    saveAdvice(advice)
 
#human-generated explanations of use of current cards
def explainCardsAdvice(player,jsonObject, advice):
    adv_text=""
    for deck in jsonObject['cards']:
        card_id = jsonObject['cards'][deck]
        card_title = str.split(pieces._CARDS[card_id]["text"],":")[0]
        exp_string = pieces._CARDS[card_id]["explain"]
        adv_text = adv_text+card_title+": "+exp_string+"\n"
    advice['advice']=adv_text
    saveAdvice(advice)
 
#whether player's home region is 'under threat' or they have already lost number 1 status there
def homeReportAdvice(player, gameState, thisHistory, jsonObject, advice):
    player_id = gameState._get_pid(player)
    home = gameState._grande_region(player_id)
    turn = gameState._get_round()
    #check for control of region, right now
    advice['advice']['control'] = (gameState._rank_region(home).get(player_id,-1) == 1)
    
    #check for recent activity from not-you
    moves = thisHistory['pieces'][home,:,turn]-thisHistory['pieces'][home,:,(turn-1)]
    advice['advice']['moves'] = [gameState._get_player_name(i) for i in range(gameState._num_players) if moves[i]>0 and i!=player_id]

    #check if it's in anyone's interest to overtake you
    targeters=[]
    for p in range(gameState._num_players):
        if p!=player_id:
            targets = reportTargets(gameState, p, stopRegion=home)
            if home in targets:
                targeters=targeters+[p]
    advice['advice']['targeters']=[gameState._get_player_name(t) for t in targeters]
    
    #check if it's in your interest to overtake someone else
    targeted=[]
    ptargets = reportTargets(gameState,player_id)
    for p in range(gameState._num_players):
        if p!=player_id:
            opp_home = gameState._grande_region(p)
            if opp_home in ptargets:
                targeted = targeted + [p]
    advice['advice']['targeted']=[gameState._get_player_name(t) for t in targeted]
    saveAdvice(advice)
 
#regions where each player has been scoring historically, whether it's first/second/third, and how many areas the player is scoring.
def opponentReportAdvice(player, gameState, thisHistory, jsonObject, advice):
    regions = pieces._NUM_EXT_REGIONS
    for r in range(regions):
        advice['advice'][pieces._REGIONS[r]]={'pieces':{},'ranks':{}}
        for p in range(gameState._num_players):
            currentPieces=gameState._board_state[r,p]
            placements=np.full(gameState._get_round()+1,0)
            ranks=[]
            trackCabs=currentPieces
            for rd in range(gameState._get_round(), 0, -1):
                placed=thisHistory['pieces'][r,p,rd] - thisHistory['pieces'][r,p,rd-1]
                if placed>0:
                    placed = min(placed,trackCabs)
                    placements[rd]=placed
                    trackCabs -= placed
                if el_grande._SCORING_ROUND[rd]:
                    ranks = ranks + [thisHistory['ranks'][r,p,rd]]
            pname=gameState._get_player_name(p)
            if currentPieces>0:
                advice['advice'][pieces._REGIONS[r]]['pieces'][pname]=placements.tolist()
            advice['advice'][pieces._REGIONS[r]]['ranks'][pname]=ranks

    saveAdvice(advice)
 
#predicted scores for each player based on this round
def scorePredictionsAdvice(player, gameState, jsonObject, advice):
    theseScores = gameState._score_all_regions()
    currentScores = gameState._current_score()
    #for each player, return their current point total and how much they'd score if this were a scoring round
    advice['advice']={gameState._players[p]:(int(currentScores[p]),int(theseScores[p])) for p in range(gameState._num_players)}
    saveAdvice(advice)
 
#suggestion of what cards to play
def suggestionAdvice(player, gameState, jsonObject, advice, saveMe=True):
    mc_sims = 800

    turn = gameState._get_round()
    advicePhase = gameState._get_current_phase_name()
    #multistep to end of all actions if we're anywhere in action phase
    if advicePhase.startswith("action"):
        advicePhase="action"
    simGame = gameState.clone()
    simGame._end_turn = 3*((turn-1)//3) + 3 #next scoring round from where we are
    rng = np.random.RandomState()
    eval=mcts.RandomRolloutEvaluator(4,rng)
    mbot = mcts.MCTSBot(simGame._game,2,mc_sims,eval,random_state=rng,solve=True,verbose=False)
    act, actList, returns = mbot.multi_step(simGame)
    actReport=[] #string actions to return as advice
    for a in actList:
        if simGame._get_current_phase_name().startswith(advicePhase): #keep suggestions to this phase, no run-ons
            actReport = actReport + [simGame.action_to_string(a,withPlayer=False)]
            simGame.do_apply_action(a)
    advice['advice'] = actReport
    
    if saveMe:
        saveAdvice(advice)

#specific alert of problems that might come up
def alertAdvice(player, gameState, thisHistory, jsonObject, advice):
    #home regions lost this turn?
    your_id = gameState._get_pid(player)
    advice_string=""
    for player_id in range(gameState._num_players):
        home = gameState._grande_region(player_id)
        turn = gameState._get_round()
        #check for control of region, right now
        hasControl = (gameState._rank_region(home).get(player_id,-1) == 1)
        if not hasControl:
            #was control just lost?
            lastControl = thisHistory["ranks"][home,player_id,(turn-1)] == 1
            if lastControl:
                #report for yourself or for opponent
                if player_id==your_id:
                    advice_string += "You have lost control of your home region.\n"
                else:
                    advice_string += gameState._players[player_id] + " has lost control of their home region.\n"
    if advice_string!="":
        advice['advice']['home']=advice_string 
    saveAdvice(advice)

#insert information about caballero movements in a way that will be
#easy to interpret for the advisor
def addRelevantGameHistory(gameState, gameHistory):

    turn = gameState._get_round()
    regions = pieces._NUM_EXT_REGIONS
    players = gameState._num_players
   
    gameHistory["pieces"][:,:,turn] = gameState._board_state[:regions,:players]
    for r in range(regions):
        #rank everything last by default
        rd=gameState._rank_region(r)
        ranks=[rd.get(i,players) for i in range(players)]
        gameHistory["ranks"][r,:,turn]=ranks


#return next n regions to be targeted by this player, using greedy algorithm
def reportTargets(gameState, player, stopRegion=-1, stopStep=5, validList=None):
    #default - return top 5 targets. Option to stop early at a region of interest
    regions = pieces._NUM_EXT_REGIONS
    testPieces = np.full(regions,0)
    retList = []

    validRegions = [r for r in range(regions)]
    if validList!=None:
        #a list of valid regions has been provided - use it
        validRegions=validList
     
    for test in range(stopStep):
        #each round, work out how best to improve the point gap
        testState = gameState.clone()
        testState._board_state[:regions,player]+=testPieces
        defaultPoints=[testState._score_one_region(r) for r in range(regions)]
        defaultPointGap = getPointGap(sum(defaultPoints),player)
        bestPointGap = defaultPointGap #floor for improvement
        bestRegion = -2 #equivalent to 'stay in court'
        
        testState._board_state[:regions,player]+=1
        testPoints=[testState._score_one_region(r) for r in range(regions)]
        for r in validRegions:
            #check the effect on the point gap of subbing in each testPoints in turn
            testArray=[testPoints[reg] if reg==r else defaultPoints[reg] for reg in range(regions)]
            pointGap = getPointGap(sum(testArray),player)
            if pointGap>bestPointGap or (pointGap==bestPointGap and r==stopRegion):
                bestPointGap=pointGap
                bestRegion=r

        #once we identified best region for this round, stop if it's the stopRegion 
        #otherwise register this as the nth best and keep going
        retList = retList + [bestRegion]

        if bestRegion==stopRegion:
            return retList
        else:
            testPieces[bestRegion]+=1

    return retList

#if player is first, pointGap is +ve or 0 distance to next player
#if player is not first, pointGap is -ve distance to top player
def getPointGap(pointArray, player):
    top = max(pointArray)
    next=0
    nextArr=[p for p in pointArray if p!=top]
    if len(nextArr)>0:
        next = max([p for p in pointArray if p!=top])
    if pointArray[player]==top:
        #if equal firsts, gap is zero
        if len(np.where(pointArray==top)[0])>1:
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

def log(message):
    try:
        now = datetime.now()
        f = open(_LOG_FILE,"a")
        f.write(now.strftime("%d/%m/%Y %H:%M:%S")+": "+message+"\n")
        f.close()
        print(message)
    except:
        print("Couldn't write to log file")
