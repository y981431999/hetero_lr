#-*- codeing = utf-8 -*-
# @Time : 2022/7/5 14:06
# @Author : 夏冰雹
# @File : reConnect.py 
# @Software: PyCharm
import zmq
import json
class Connect_gennerator:
    path = 'config.json'
    f = open(path, 'r', encoding='utf-8')
    m = json.load(f)
    ip = m["connect"]["ip"]
    port = m["connect"]["port"]
    @staticmethod
    def getServerConnect():
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind('tcp://*:'+Connect_gennerator.port)
        return socket
    @staticmethod
    def getClientConnect():
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        ip = 'tcp://'+Connect_gennerator.ip+':'+Connect_gennerator.port
        socket.connect(ip)
        return socket