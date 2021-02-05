import el_grande
import json
import couchdb

gamedocid='d16e7ab43670c7656101aa0a30098186'
gamehistids=['d16e7ab43670c7656101aa0a30098ceb','d16e7ab43670c7656101aa0a30099463','d16e7ab43670c7656101aa0a30099f6e','d16e7ab43670c7656101aa0a3009af36','d16e7ab43670c7656101aa0a3009bc1f','d16e7ab43670c7656101aa0a3009c5bd','d16e7ab43670c7656101aa0a3009d43e','d16e7ab43670c7656101aa0a3009de47','d16e7ab43670c7656101aa0a3009e939','d16e7ab43670c7656101aa0a3009f803','d16e7ab43670c7656101aa0a300a0299','d16e7ab43670c7656101aa0a300a1287','d16e7ab43670c7656101aa0a300a2145']

egame = el_grande.ElGrandeGame()

couchip = '127.0.0.1:5984'
credentials = 'admin:elderberry'
couch = couchdb.Server('http://'+credentials+'@'+couchip)
gamedb = couch['game']
gamehistdb = couch['game_history']

#check out the test data
gameitem = gamedb[gamedocid]
for id in gamehistdb:
    egstate = egame.new_initial_state()
    egstate._load_game_info(gameitem)
    item = gamehistdb[id]
    if item.get('name','')==gameitem['Name']:
        egstate._load_game_state(item)
    
    
