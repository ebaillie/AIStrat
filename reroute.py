import base64
import logging
import os
import re
import requests
import sys
import advisor

from flask import Flask,jsonify,redirect,request
from flask import request
from multiprocessing import Process, Manager

app = Flask(__name__)

localhost='127.0.0.1'

manager = Manager()
game_histories = manager.dict()

@app.route('/')
def hello_world():
    return 'Tiny TTS rerouter is here for you'


@app.route('/forward',methods=['PUT'])
def forward_request():

    params = request.get_json(force=True)

    url = params.get('url')
    method = params.get('method')

    if not url:
        return ('URL not specified',422)

    if url.find(localhost)==-1:
        return ('Nup, not goin there',422)

    if not method:
        return ('HTTP method not specified',422)

    headers = {
            **(params.get('headers') or {}),
                }

    headers['Content-Type'] = 'application/json'

    body = None

    if 'body' in params:
        body = base64.b64decode(params['body']) if params.get('base64') else params['body']
    print("forwarded body ",body,"\n")

#    if body:
#        body = body.encode()

    #send the body to the advisor code, as well as storing it in the db
    advisor.processDBInput(body, game_histories)
    response = requests.request(method, url, timeout=60.0, headers=headers, data=body)

    wrapped_response = {
        'status_code': response.status_code,
        'headers': dict(response.headers),
    }

    if response.content:
        if 'content-type' in response.headers and response.headers['content-type'].lower() == 'application/octet-stream':
            wrapped_response['body'] = base64.b64encode(response.content)
            wrapped_response['base64'] = True
        else:
            wrapped_response['body'] = response.text

    return jsonify(wrapped_response)
