#-*- codeing = utf-8 -*-
# @Time : 2022/7/28 14:08
# @Author : 夏冰雹
# @File : 02_RDD_create_textFile.py 
# @Software: PyCharm
from paillier_en.paillier import PaillierKeypair
from pyspark.sql import SparkSession
from pyspark import SparkConf,SparkContext
import numpy as np
import pickle
from optim.Weights import LinearModelWeights
import reConnect
if __name__ == '__main__':
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    socket = reConnect.Connect_gennerator.getServerConnect()
    role = 'host'
    def send_rdd(data):
        if role == 'host':
            rec = socket.recv()
            socket.send(data)
        else:
            socket.send(data)
            rec = socket.recv()
        rec = pickle.loads(rec)
        return rec
    guest_rdd = sc.parallelize([],8)
    while True:
        data = send_rdd(pickle.dumps(0))
        if data == -1:
            break
        else:
            rdd = sc.parallelize([data])
            guest_rdd = guest_rdd.union(rdd)
    print(guest_rdd.collect())
    print(guest_rdd.count())

