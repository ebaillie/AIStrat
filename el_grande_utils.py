#utility functions for extracting metrics and information from el_grande game or game state
import numpy as np
import json
import el_grande
import simple_grande
import castillo_game
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python import rl_environment
import random
import mcts_ext as mcts

class RegionBandit:
  """Class for running multi-arm bandit evaluations of simultaneous secret region choices"""

  def __init__(self, arm_count,epsilon):
    self.arm_count=arm_count
    self.arm_vals = np.zeros(arm_count) # cumulative values
    self.arm_choices = np.full(arm_count,0.001) # cumulative counts,with small eps for div-by-zero
    self.epsilon=epsilon

  def update(self,arm,val):
    self.arm_choices[arm] += 1
    self.arm_vals[arm]+=val

  def select(self,embargo=[],with_eps=True):
    choices=[a for a in range(self.arm_count) if not a in embargo]
    eps_rand=random.random()
    if eps_rand<self.epsilon and with_eps:
      return random.choice(choices)
    else:
      meanvals=[self.arm_vals[c]/self.arm_choices[c] for c in choices]
      best_choices=[choices[c] for c in range(len(choices)) if meanvals[c]==max(meanvals)]
      return random.choice(best_choices)

  def reward(self,arm):
    return self.arm_vals[arm]/self.arm_choices[arm]

  def mean_reward(self):
    return np.sum(self.arm_vals)/np.sum(self.arm_choices)

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

def cabs_to_play(state,pl):
  pl=internal_rep(state,pl)
  #how many caballeros will be available by the time action card is played?
  power_list = np.where(state._pcard_state & pow(2,pl) == pow(2,pl))[0]
  assert(len(power_list)==1)
  move_cabs=min(state._board_state[state._game._province_idx,pl],state._game._power_cabs[power_list[0]])
  return(int(state._board_state[state._game._court_idx,pl]+move_cabs))

def cards_out(state):
  cards=np.append(np.where(state._acard_round==state._get_round()),[42])[:5] 
  return[state._game._cardtrack[c] for c in cards]
 
def board_cabs(state,player):
  #how many total caballeros does this player have on the board?
  player=internal_rep(state,player)
  return int(sum(state._board_state[:state._game._num_ext_regions,player]))

def playMetaFromState(estate,qagent,stopAfterPower=False):
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
    #translate actions back to core el grande action, for reporting back
    eg_decks=np.append(np.where(estate._acard_round==sgs._round),[42])[:5]
    #note - can make this close to deterministic by changing is_evaluation to True
    agent_output =qagent.step(time_step, is_evaluation=False)
    action=agent_output.action
    time_step = env.step([action])
    if action<simple_grande._POWER_CARDS:
      actions+=[(player_id,int(action+el_grande._ACT_POWERS))]
    else:
      if stopAfterPower:
        return(actions)
      actions+=[(player_id,int(eg_decks[action-simple_grande._POWER_CARDS]+el_grande._ACT_CARDS))]
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
  new_idxs=castillo_scoring_order(state,player)
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

def predictCastilloMoves(state,player=-1):
  #predict likely distribution of player/opponent moves in castillo/scoring round
  player=internal_rep(state,player)
  advice={}
  if state._board_state[state._game._castillo_idx,player]==0:
    return advice 
  cabMovers = [p for p in range(state._num_players) if state._board_state[state._game._castillo_idx,p]>0]
  if len(cabMovers)<=1:
    return advice

  sim_count=10
  sim_iterations=100

  choices=np.full((state._num_players,state._num_players*sim_count),0)
  for sim_player in range(state._num_players):
    temp_choices = np.full((state._num_players,sim_count),0)
    cg=castillo_game.CastilloGame({"state":castillo_game_string(state,sim_player)})
    cgs=cg.new_initial_state()
    rng=np.random.RandomState()
    eval=mcts.RandomRolloutEvaluator(1,rng)
    mbot = mcts.MCTSBot(cg,2,sim_iterations,eval,random_state=rng,solve=True,verbose=False)
    for i in range(sim_count):
      c,cc,vc,_=mbot.multi_step(cgs,"all") #do sim runs, reporting back whole game history
      #pad out truncated multi-steps, where not all players have castillo pieces
      optpad=np.full(state._num_players-len(cc),9).tolist() 
      temp_choices[:,i]=cc+optpad
    #convert back to indexing by core game player numbers
    idxs=castillo_scoring_order(state,sim_player)
    for p in idxs:
      choices[p,sim_player*sim_count:(sim_player+1)*sim_count]=temp_choices[idxs[p]]

  rankvals=np.full(state._game._num_regions,0.0)
  scorevals=np.full(state._game._num_regions,0.0)
  #for each set of choices, calculate the payoff for the selected player in terms of raw score and rank improvement 
  startscores=state._score_all_regions()
  rankarray=[]
  for it in range (state._game._num_regions):
    if it==state._king_region():
      rankvals[it]=-1000
      continue
    for tries in range(sim_count): 
      tempstate=state.clone()
      tempchoice=np.full(state._num_players,0)
      for p in range(state._num_players):
        if p==player:
          tempchoice[p]=it
          tempstate._quickmove_cabs(state._game._castillo_idx,it,-1,p)
        else:
          tempchoice[p]=random.choice(choices[p,:])
          tempstate._quickmove_cabs(state._game._castillo_idx,tempchoice[p],-1,p)
      newscores=tempstate._score_all_regions()
      newrank=round(state._scores_as_margins(startscores+newscores)[player],2)
      rankarray.append((list(tempchoice),newscores[player],newrank))
      rankvals[it]+=round(state._scores_as_margins(startscores+newscores)[player],2)
      scorevals[it]+=tempstate._score_one_region(it)[player]

  rank_expect=rankvals/sim_count
  score_expect=scorevals/sim_count 
  #format as playerid:{region number:proportion} for relevant players
  for pl in cabMovers:
    unique, counts = np.unique(choices[pl,:], return_counts=True)
    pcount = dict(zip(unique.tolist(), (counts/sum(counts)).tolist()))
    advice[pl] = {p:float(pcount[p]) for p in pcount}
  return advice,score_expect.tolist(),rank_expect.tolist()

def makeBanditRegionExclusions(state,cardName):
  king_reg=state._king_region()
  if cardName=="Deck4_Special":
    return {p:[king_reg] for p in range(state._num_players)}
  elif cardName=="Deck2_Province":
    excl={}
    for p in range(state._num_players):
      testexcl=list(set([king_reg]+[r for r in range(state._game._num_regions) if state._region_cabcount(r,p)<2]))
      if len(testexcl)==state._game._num_regions:
        testexcl=list(set([king_reg]+[r for r in range(state._game._num_regions) if state._region_cabcount(r,p)<1]))
      if len(testexcl)==state._game._num_regions:
        testexcl=[king_reg]
      excl[p]=testexcl
    return excl 
  elif cardName=="Deck2_Provinceall":
    excl={}
    for p in range(state._num_players):
      testexcl=list(set([king_reg]+[r for r in range(state._game._num_regions) if state._region_cabcount(r,p)<1]))
      if len(testexcl)==state._game._num_regions:
        testexcl=[king_reg]
      excl[p]=testexcl
    return excl 
  elif cardName=="Deck4_Eviction":
    return {p:[king_reg] for p in range(state._num_players)}

def runBandits(state,bandit_array,exclusions,start_scores,cardName):
  estate=state.clone()
  choices=np.full(estate._num_players,0)
  if cardName=="Deck4_Special":
    for p in range(estate._num_players):
      choices[p]=bandit_array[p].select(embargo=exclusions[p])
      estate._set_secret_region(choices[p],p)
    estate._unique_score()
    values=estate._scores_as_margins(estate._current_score())
  elif cardName=="Deck2_Province":
    for p in range(estate._num_players):
      choices[p]=bandit_array[p].select(embargo=exclusions[p])
    for p in range(estate._num_players):
      fromreg=int(choices[p])
      estate._quickmove_cabs(fromreg,estate._game._province_idx,2,p)
    values=estate._scores_as_margins(estate._score_all_regions()+start_scores)
  elif cardName=="Deck2_Provinceall":
    for p in range(estate._num_players):
      choices[p]=bandit_array[p].select(embargo=exclusions[p])
    for p in range(estate._num_players):
      fromreg=int(choices[p])
      estate._quickmove_cabs(fromreg,estate._game._province_idx,-1,p)
    values=estate._scores_as_margins(estate._score_all_regions()+start_scores)
  elif cardName=="Deck4_Eviction":
    fromreg=estate._secret_region(estate._rsp_player)
    choices[estate._rsp_player]=fromreg
    for p in range(state._num_players):
      if p!=estate._rsp_player:
        choices[p]=bandit_array[p].select(embargo=exclusions[p])
    for p in range(state._num_players):
      if p!=estate._rsp_player:
        toreg=int(choices[p])
        estate._quickmove_cabs(fromreg,toreg,-1,p)
    values=estate._scores_as_margins(estate._score_all_regions()+start_scores)
  elif cardName=="scoring":
    castillo_score=estate._score_one_region(estate._game._castillo_idx)
    for p in range(estate._num_players):
      choices[p]=bandit_array[p].select(embargo=exclusions[p])
    for p in range(state._num_players):
      estate._quickmove_cabs(estate._game._castillo_idx,int(choices[p]),-1,p)
    values=estate._scores_as_margins(estate._score_all_regions()+start_scores+castillo_score)
    

  return choices,values

def banditScoring(state,pl):
  """Multi-arm Bandit code to generate predictions and explanation info for the scoring round
     """
  pl=internal_rep(state,pl)
  bandit_runs=2000
  eval_runs=100
  eps=0.1
  bandit_array=[RegionBandit(state._game._num_regions,eps) for i in range(state._num_players)]
  start_scores=state._current_score()
  choices=np.full(state._num_players,0)
  exclusions={p:[state._king_region()] for p in range(state._num_players)}
  for i in range(bandit_runs):
    choices,values = runBandits(state,bandit_array,exclusions,start_scores,"scoring")
    for p in range(state._num_players):
      bandit_array[p].update(choices[p],values[p])
  rv=[bandit_array[b].select(embargo=exclusions[b],with_eps=False) for b in range(state._num_players)]
  rewards={r:bandit_array[pl].reward(r) for r in range(state._game._num_regions) if r not in exclusions[pl]}
  predictions={p:rv[p] for p in range(state._num_players) if p!=pl}
  #do some evaluation runs to see how much variance there is in the prediction
  player_region=rv[pl]
  exclusions[pl]=[r for r in range(state._game._num_regions) if r!=player_region]
  player_rewards=[]
  for i in range(eval_runs):
    choices,values = runBandits(state,bandit_array,exclusions,start_scores,"scoring")
    player_rewards.append(values[pl])
  return rv,rewards,predictions,float(np.std(player_rewards))

def banditResponse(state,pl,cardName):
  """Multi-arm Bandit code to generate response predictions and explanation info for the simultaneous-choice
     response situations in El Grande: Eviction (choose region to move to),Province and Provinceall (choose region
     to evict from), and Unique Score"""
  pl=internal_rep(state,pl)
  bandit_runs=2000
  eval_runs=100
  eps=0.1
  bandit_array=[RegionBandit(state._game._num_regions,eps) for i in range(state._num_players)]
  start_scores=state._current_score()
  choices=np.full(state._num_players,0)
  exclusions=makeBanditRegionExclusions(state,cardName)
  for i in range(bandit_runs):
    choices,values = runBandits(state,bandit_array,exclusions,start_scores,cardName)
    for p in range(state._num_players):
      bandit_array[p].update(choices[p],values[p])
  rv=[bandit_array[b].select(embargo=exclusions[b],with_eps=False) for b in range(state._num_players)]
  rewards={r:bandit_array[pl].reward(r) for r in range(state._game._num_regions) if r not in exclusions[pl]}
  predictions={p:rv[p] for p in range(state._num_players) if p!=pl}
  #do some evaluation runs to see how much variance there is in the prediction
  player_region=rv[pl]
  exclusions[pl]=[r for r in range(state._game._num_regions) if r!=player_region]
  player_rewards=[]
  for i in range(eval_runs):
    choices,values = runBandits(state,bandit_array,exclusions,start_scores,cardName)
    player_rewards.append(values[pl])
  return rv,rewards,predictions,float(np.std(player_rewards))


def getContentions(state,pl):
  pl=internal_rep(state,pl)
  contentions=0
  for r in range(state._game._num_regions):
    if state._board_state[r,pl]>0:
      contentions+=len(np.where(state._board_state[r,:state._num_players]==state._board_state[r,pl])[0])-1
  return contentions

def getContentionRegions(state,pl):
  pl=internal_rep(state,pl)
  contentions=np.full(state._game._num_regions,-1)
  for r in range(state._game._num_regions):
    if state._board_state[r,pl]>0:
      contentions[r]=len(np.where(state._board_state[r,:state._num_players]==state._board_state[r,pl])[0])-1
  if max(contentions)>0:
    return np.where(contentions==max(contentions))[0].tolist()
  else:
    return []


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

def getMinCabcountRegionVals(state,pl,withCastillo=False):
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
    defaultPointGap = _getPointGap(sum(defaultPoints),pl)
    bestPointGap = defaultPointGap #floor for improvement
    bestRegion = -2 #equivalent to 'stay in court'
        
    testState._board_state[:regions,pl]+=1
    testPoints=[testState._score_one_region(r) for r in range(regions)]
    for r in validRegions:
      #check the effect on the point gap of subbing in each testPoints in turn
      testArray=[testPoints[reg] if reg==r else defaultPoints[reg] for reg in range(regions)]
      pointGap = _getPointGap(sum(testArray),pl)
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
    if len(targets)==0:
      continue
    if len(targets)<cabsAvail:
      targets=targets+[random.choice(targets) for i in range(cabsAvail-len(targets))]
    targets=targets[:cabsAvail]
    for t in targets:
      testState._board_state[t,pl]+=1
    changes=testState._scores_as_margins(testState._score_all_regions())
    pointGaps[r]=changes[pl]-initMargin[pl]
  
  bestValue=max(pointGaps) 
  return(np.where(pointGaps==bestValue)[0],int(bestValue)) 

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
  bestValue=max(pointGaps.reshape(regions*scbds,1))[0]
  scbdregs=np.where(pointGaps==bestValue)
  whichregs=scbdregs[0]
  whichboards=scbdregs[1]
  return(whichregs,whichboards,int(bestValue))

def _getPointGap(pointArray, player):
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

def stableRegions(bfr,aft,pl):
  #score regions according to how stable they are before and after wrt player
  #1 = fully stable, 0.5=some change but no rank change or moved grande/king, 0=rank changed or grande/king movement

  pl=internal_rep(bfr,pl)
  regscores=np.full(bfr._game._num_ext_regions,0.0)
  gk=el_grande._ST_BDY_GRANDE_KING
  for r in range(bfr._game._num_ext_regions):
    if all(bfr._board_state[r,0:gk+1]==aft._board_state[r,0:gk+1]):
      regscores[r]=1
    elif bfr._rank_region(r).get(pl,4)==aft._rank_region(r).get(pl,4) and bfr._board_state[r,gk]==aft._board_state[r,gk]:
      #some change but rank and g/k okay
      regscores[r]=0.5

  return regscores.tolist()
