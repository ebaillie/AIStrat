#utility functions for extracting metrics and information from el_grande game or game state
import numpy as np
import el_grande
import simple_grande
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python import rl_environment

def internal_rep(state,pl):
  if type(pl)==str:
    pl=state._players.index(pl)
  if pl==-1:
    pl=state._cur_player
  return pl

def this_turn_position(state,pl):
  #position in the current order
  pl=internal_rep(state,pl)
  pc=state._get_power_card(pl)
  if len(pc)==0:
    return(-1)
  pc_hi=[p for p in np.where(state._pcard_state !=0)[0] if p>pc]
  return len(pc_hi)+1

def next_turn_position(state,pl):
  #position in order for next round
  pcs=np.where(state._pcard_state !=0)[0]
  if len(pcs)==0:
    return(-1)
  startplayer = int(np.log2(state._pcard_state[pcs[0]]))
  return((pl-startplayer)%(state._num_players)+1)

def cards_out(state):
  cards=np.append(np.where(state._acard_round==state._get_round()),[42])[:5] 
  return[state._game._cardtrack[c] for c in cards]
 
def board_cabs(state,player):
  #how many total caballeros does this player have on the board?
  player=internal_rep(state,player)
  return int(sum(state._board_state[:state._game._num_ext_regions,player]))

def playMetaFromState(estate,qagent):
  #returns playerid/el_grande action pairs for projected power and action phases,
  #skipping other phases

  sg=simple_grande.SimpleGrandeGame()
  sgs=sg.new_initial_state()
  sgs._load_from_eg(estate)
  env = rl_environment.Environment("simple_grande")
  env.reset()
  env.set_state(sgs)
  time_step=env.get_time_step()
  actions=[]
  while not time_step.last():
    player_id = time_step.observations["current_player"]
    qagent._player_id=player_id
    #note - can make this close to deterministic by changing is_evaluation to True
    agent_output =qagent.step(time_step, is_evaluation=False)
    action=agent_output.action
    time_step = env.step([action])
    #translate actions back to core el grande action, for reporting back
    eg_decks=np.append(np.where(estate._acard_round==sgs._round),[42])[:5]
    if action<simple_grande._POWER_CARDS:
      actions+=[(player_id,action+el_grande._ACT_POWERS)]
    else:
      actions+=[(player_id,eg_decks[action-simple_grande._POWER_CARDS]+el_grande._ACT_CARDS)]
  return(actions)

def castillo_scoring_order(state,player=-1):
  #order in which players will do their (secret) castillo choice - needed for castillo subgame
  player=internal_rep(state,player)
  so=state._playersleft+state._playersdone
  zero_point=so.index(player)
  new_idxs={v:(so.index(v)-zero_point)%len(so) for v in so}
  return new_idxs

def castillo_game_string(state,player=-1):
  player=internal_rep(state,player)
  #translate game state into CastilloGame format, for running tiny sims
  state_vals={"players":state._num_players,"rewards":state._points,"king":state._king_region()+1}
  #put player state into castillo game state in order, starting from current player
  #region 0 is the castillo
  board=np.full(state._num_players*state._game._num_ext_regions,0)
  grandes=np.full(state._num_players,0)
  scores=np.full(state._num_players,0)
  new_idxs=castillo_scoring_order(player)
  for p in new_idxs:
    idx=new_idxs[p]
    for r in range(state._game._num_regions):
      board[(r+1)*state._num_players+idx] = state._board_state[r,p]
    #castillo
    board[idx] = state._board_state[state._game._castillo_idx,p]
    grandes[idx] = state._grande_region(p)+1
    scores[idx] = state._current_score()[p]
  state_vals["board"]="".join([chr(ord('A')+b) for b in board])
  state_vals["grandes"]=grandes.tolist()
  state_vals["scores"]=scores.tolist()
  return json.dumps(state_vals)

def getContentions(state,pl):
  pl=internal_rep(state,pl)
  contentions=0
  for r in range(state._game._num_regions):
    if state._board_state[r,pl]>0:
      contentions+=len(np.where(state._board_state[r,:state._num_players]==state._board_state[r,pl])[0])-1
  return contentions

def getMaxCabcountRegions(state,withCastillo=False):
  regions=state._game._num_regions
  if withCastillo:
    regions+=1
  cabcount= np.sum(state._board_state[:regions,:state._num_players],axis=1)
  return(np.where(cabcount==max(cabcount))[0])

def getMaxCabcountRegionVals(state,pl,withCastillo=False):
  regions=state._game._num_regions
  pl=internal_rep(state,pl)
  if withCastillo:
    regions+=1
  cabcount= np.sum(state._board_state[:regions,:state._num_players],axis=1)
  keyRegions=np.where(cabcount==max(cabcount))[0]
  vals=np.zeros(state._num_players)
  for r in keyRegions:
    vals+=state._score_one_region(r)
  return(keyRegions,state._scores_as_margins(vals)[pl])

def getMinCabcountRegionVals(state,withCastillo=False):
  regions=state._game._num_regions
  pl=internal_rep(state,pl)
  if withCastillo:
    regions+=1
  cabcount= np.sum(state._board_state[:regions,:state._num_players],axis=1)
  keyRegions=np.where(cabcount==min(cabcount))[0]
  vals=np.zeros(state._num_players)
  for r in keyRegions:
    vals+=state._score_one_region(r)
  return(keyRegions,state._scores_as_margins(vals)[pl])

def getMaxCabcountRegionsPlayer(state,pl,withCastillo=False):
  pl=internal_rep(state,pl)
  regions=state._game._num_regions
  if withCastillo:
    regions+=1
  return np.where(state._board_state[:regions,pl]==max(state._board_state[:regions,pl]))[0]

def getTargets(state, pl, stopRegion=-1, stopStep=5, validList=None):
  pl=internal_rep(state,pl)
  #default - return top 5 targets. Option to stop early at a region of interest
  regions = state._game._num_ext_regions
  testPieces = np.full(regions,0)
  retList = []

  validRegions = [r for r in range(regions)]
  if validList!=None:
    #a list of valid regions has been provided - use it
    validRegions=validList
     
  for test in range(stopStep):
    #each round, work out how best to improve the point gap
    testState = state.clone()
    testState._board_state[:regions,pl]+=testPieces
    defaultPoints=[testState._score_one_region(r) for r in range(regions)]
    defaultPointGap = getPointGap(sum(defaultPoints),pl)
    bestPointGap = defaultPointGap #floor for improvement
    bestRegion = -2 #equivalent to 'stay in court'
        
    testState._board_state[:regions,pl]+=1
    testPoints=[testState._score_one_region(r) for r in range(regions)]
    for r in validRegions:
      #check the effect on the point gap of subbing in each testPoints in turn
      testArray=[testPoints[reg] if reg==r else defaultPoints[reg] for reg in range(regions)]
      pointGap = getPointGap(sum(testArray),pl)
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

def getBestKingRegions(state,pl,cabsAvail=5,validList=None):
  pl=internal_rep(state,pl)
  regions = state._game._num_regions
  
  pointGaps=np.full(regions,-1)
  initMargin=state._scores_as_margins(state._score_all_regions())
  validRegions = [r for r in range(regions)]
  if validList!=None:
    #a list of valid regions has been provided - use it
    validRegions=validList

  targetList=getTargets(state,pl,stopStep=cabsAvail*2)
  for r in validRegions:
    testState = state.clone()
    testState._move_king(r)
    targets=[t for t in targetList if t in testState._game._neighbors[r]]
    if len(targets)<cabsAvail:
      targets=targets+[random.choice(targets) for i in range(cabsAvail-len(targets))]
    targets=targets[:cabsAvail]
    for t in targets:
      testState._board_state[t,pl]+=1
    changes=testState._scores_as_margins(testState._score_all_regions())
    pointGaps[r]=changes[pl]-initMargin[pl]
  
  bestValue=max(pointGaps) 
  return(np.where(pointGaps==bestValue),bestValue) 

def getBestScbdRegions(state,pl):
  pl=internal_rep(state,pl)
  regions=state._game._num_ext_regions
  scbds=2
  pointGaps=np.full((regions,scbds),-1)
  initMargin=state._scores_as_margins(state._score_all_regions())
  for s in range(scbds):
    testState=state.clone()
    for r in range(regions):
      testState._move_scoreboard(s,r)
      changes=testState._scores_as_margins(testState._score_all_regions())
      pointGaps[r,s]=changes[pl]-initMargin[pl]
  bestValue=max(pointGaps.reshape(regions*scbds,1))
  scbdregs=np.where(pointGaps==bestValue)
  whichregs=scbdregs[0]
  whichboards=scbdregs[1]
  return(whichregs,whichboards,bestValue)
