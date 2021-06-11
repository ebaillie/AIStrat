import el_grande
import el_grande_pieces as pieces
import json
import couchdb
import time
import datetime

couchip = '127.0.0.1:5984'
credentials = 'admin:elderberry'
couch = couchdb.Server('http://'+credentials+'@'+couchip)
gamedb = couch['game']
gamehistdb = couch['game_history']
gamelogdb = couch['game_log']
advarchivedb = couch['game_advice_archive']

def gamesForDate(y,m,d):
    # find all games on given day, by name/time identifier
    dt=datetime.datetime(year=y,month=m,day=d)
    starttime=time.mktime(dt.timetuple())
    #1 day is 86400 seconds
    return([(gamedb[g]['Name'],gamedb[g]['Time'],gamedb[g]['Players']) for g in gamedb if 
                float(gamedb[g].get('Time',0))>starttime and 
                float(gamedb[g]['Time'])<(starttime+86400)])

def gameHistIDs(game,starttime):
    return( [g for g in gamehistdb if 
            gamehistdb[g].get('name','')==game and 
            gamehistdb[g].get('time','')==starttime])

def gameAdviceIDs(game,starttime):
    return( [g for g in advarchivedb if 
            advarchivedb[g].get('game','')==game and 
            advarchivedb[g].get('time','')==starttime])

def gameLogIDs(game,starttime):
    return( [g for g in gamelogdb if 
            gamelogdb[g].get('game','')==game and 
            gamelogdb[g].get('time','')==starttime])

def advisorAttentionReport(logrecords,player):
    #which advisor panels has player been looking at, and at what point?
    lastrecord=None
    thisrecord={'player':player}
    outstring="player,viewtime,viewing,round,phase,switch,duration\n"
    for rec in [gamelogdb[l] for l in logrecords if 
                gamelogdb[l].get('player','')==player and gamelogdb[l].get('switch','')!='']:
        #assume time-ordered
        thisrecord['viewtime']=round(float(rec['viewtime']) - float(rec['time'])) 
        thisrecord['viewing']=rec['viewing'][:-1]
        thisrecord['round']=rec['round']
        thisrecord['phase']=rec['phase']
        thisrecord['switch']=rec['switch']
        if lastrecord!=None:
            lastrecord["duration"]=thisrecord['viewtime'] - lastrecord['viewtime']  
            outstring+= ",".join([str(v) for v in lastrecord.values()])+"\n"
        lastrecord=thisrecord.copy()
    return outstring

def suggestionAttentionTimes(logrecords,playerlist):
    #dict of times players were looking at action and power suggestions, per player
    dTimes={}
    for player in playerlist:
        dTimes[player]={'action':{},'power':{}}
        lastrecord=None
        thisrecord={'player':player}
        for rec in [gamelogdb[l] for l in logrecords if 
                gamelogdb[l].get('player','')==player and gamelogdb[l].get('switch','')!='']:
            #assume time-ordered
            thisrecord['viewtime']=round(float(rec['viewtime']) - float(rec['time']))
            thisrecord['viewing']=rec['viewing'][:-1]
            thisrecord['round']=rec['round']
            thisrecord['phase']=rec['phase']
            if lastrecord!=None:
                rnd=lastrecord['round']
                lastrecord["duration"]=thisrecord['viewtime'] - lastrecord['viewtime']
                if lastrecord['viewing']=='power_suggestion':
                    dTimes[player]['power'][rnd] = dTimes[player]['power'].get(rnd,0)+lastrecord['duration'] 
                elif lastrecord['viewing']=='action_suggestion':
                    dTimes[player]['action'][rnd] = dTimes[player]['action'].get(rnd,0)+lastrecord['duration'] 
            lastrecord=thisrecord.copy()
    return dTimes

def advisorRequestReport(logrecords, player):
    #what sort of advice did player request, in what round?
    outstring="player,advicetype,round,phase,requestdata\n"
    for rec in [gamelogdb[l] for l in logrecords if 
            gamelogdb[l]['player']==player and gamelogdb[l].get('advicetype','')!='']:
        outstring+="{0},{1},{2},{3},{4}\n".format(rec['player'],rec['advicetype'],
                                        rec['round'],rec['phase'],rec['requestdata'])
    return outstring

def advisorCardsReport(advrecords,gamehistrecords,suggestionAttention):
    #power cards and action cards advised, compared with actual actions taken
    cardAdvice={}
    histStatus={}
    allPlayers=gamehistdb[gamehistrecords[0]]['players']
    outputstring="round,phase,player,cardAdvice,card,piecediff,king,attentionTime\n"
    for adv in advrecords:
        rec=advarchivedb[adv]
        if rec['advicetype']=='action_suggestion' or rec['advicetype']=='power_suggestion':
            cardAdvice[(rec['round'],rec['phase'],rec['player'])]=rec['advice']

    histStatus[(0,'scoring','')]={'round':0,'phase':'scoring'}
    for hist in gamehistrecords:
        rec=gamehistdb[hist]
        thisround=rec['turninfo']['round']
        thisphase=rec['turninfo']['phase']
        if thisphase=='start' or rec['turninfo']['playersleft']==[]:
            continue

        thisplayer=rec['turninfo']['playersleft'][0]
        
        thisHistRec=histStatus.get((thisround,thisphase,thisplayer),{})
        if thisphase!='scoring' and len(rec['turninfo']['playersleft'])>1:
            nextplayer=rec['turninfo']['playersleft'][1]
            thisHistRec['nextstep']=(thisround,thisphase,nextplayer)
        thisHistRec['startpieces']=rec['pieces']
        thisHistRec['startking']=rec['king']
        thisHistRec['round']=thisround
        thisHistRec['phase']=thisphase
        thisHistRec['player']=thisplayer
        thisHistRec['numplayersleft']=len(rec['turninfo']['playersleft'])
        thisHistRec['histid']=hist
        histStatus[(thisround,thisphase,thisplayer)]=thisHistRec

    #second pass - go through all the records and reconcile last/next steps
    for hskey in histStatus:
        hs=histStatus[hskey]
        nhskey=hs.get('nextstep',-1)
        if nhskey==-1:
            nextphase='power'
            nextround=hs['round']
            if hs['phase']=='power':
                nextphase='action'
            elif hs['phase']=='action':
                nextphase='scoring'
            elif hs['phase']=='scoring':
                nextround=nextround+1
            remains=0
            player=''
            for p in allPlayers:
                rec=histStatus.get((nextround,nextphase,p),{})
                num=rec.get('numplayersleft',-1)
                if num>remains:
                    remains=num
                    player=p
            nhskey=(nextround,nextphase,player) 
            hs['nextstep']=nhskey
        nhs=histStatus.get(nhskey,-1)
        if nhs!=-1:
            hs['endpieces']=nhs['startpieces']
            hs['endking']=nhs['startking']
            nextrec=gamehistdb[nhs['histid']]
            if hs['phase']=='power':
                hs['card']=nextrec['turninfo']['powercards'][hs['player']]
            elif hs['phase']=='action':
                actguid=nextrec['turninfo']['actioncards'][hs['player']]
                hs['card']=pieces._CARDS[actguid]['name'] 
        histStatus[hskey]=hs


    #now go through histStatus from start to finish, determining piece movement
    #nominal start is round 0, scoring phase, player ''
    #output relevant information

    histEl=histStatus[(0,'scoring','')]
    histEl=histStatus[histEl['nextstep']]
    while histEl!={}:
        piecediff={}
        for pl in histEl.get('endpieces',[]):
            tpd={}
            for r in pieces._REGIONS:
                st=histEl['startpieces'][pl].get(r,0)
                nd=histEl['endpieces'][pl].get(r,0)
                if (nd-st)!=0:
                    tpd[r]=(nd-st)
            if histEl['startpieces'][pl]['grande']!=histEl['endpieces'][pl]['grande']:
                tpd['grande']=histEl['endpieces'][pl]['grande']
            if tpd!={}:
                piecediff[pl]=tpd
        cardAdvString=cardAdvice.get((histEl['round'],histEl['phase'],histEl['player']),[])
        attTime=0
        if histEl['phase']=='power' or histEl['phase']=='action':
            attTime=suggestionAttention[histEl['player']][histEl['phase']].get(histEl['round'],0)
        outputstring+='{0},{1},{2},"{3}",{4},"{5}",{6},{7}\n'.format(histEl['round'],
                        histEl['phase'],histEl['player'],cardAdvString,
                        histEl.get('card',''),piecediff,histEl.get('endking',''),attTime)       
        histEl=histStatus.get(histEl.get('nextstep',''),{})
    return (outputstring)

