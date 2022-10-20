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
    o_port = m["connect"]["o_port"]
    @staticmethod
    def getServerConnect(o_port=False):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        if o_port:
            port = Connect_gennerator.o_port
        else:
            port = Connect_gennerator.port
        socket.bind('tcp://*:'+port)
        return socket
    @staticmethod
    def getClientConnect(o_port = False):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        if o_port:
            port = Connect_gennerator.o_port
        else:
            port = Connect_gennerator.port
        ip = 'tcp://'+Connect_gennerator.ip+':'+port
        socket.connect(ip)
        return socket