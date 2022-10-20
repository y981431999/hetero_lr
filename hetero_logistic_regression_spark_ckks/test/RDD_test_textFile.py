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
import reConnect
from optim.Weights import LinearModelWeights
if __name__ == '__main__':
    role = 'guest'
    n_length = 1024
    public_key, privite_key = PaillierKeypair.generate_keypair(n_length)
    conf = SparkConf().setAppName("test").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()
    file_rdd = sc.textFile("/tmp/pycharm_project_544/hetero_logistic_regression/data/breast_hetero_guest.csv")
    header = file_rdd.first().split(',')
    file_rdd = file_rdd.filter(lambda x: x[0] != header[0][0])
    def pre_proc_data(data):
        d_l = data.strip().split(',')
        key = int(d_l[0])
        value = list(map(float,d_l[1:]))
        return (key,(value))
    file_rdd = file_rdd.map(pre_proc_data)
    # Y = file_rdd.filter(lambda x:(x[0],x[1][0]))
    # X = file_rdd.filter(lambda x:(x[0],(x[1][1:])))
    model_weights = LinearModelWeights(np.array(([0] * 10)), False)
    model_weights_b = sc.broadcast(model_weights)
    model_weights_v = model_weights_b.value
    pkb,skb =sc.broadcast(public_key),sc.broadcast(privite_key)
    pk,sk = pkb.value,skb.value
    count = file_rdd.count()

    #分batch
    batch_size = 64
    id_list = file_rdd.map(lambda x:x[0]).collect()
    batch_index = []
    l = 0
    while True:
        if l>=len(id_list):
            break
        r = l+batch_size
        if r>=len(id_list):
            r = len(id_list)
        batch_index.append(id_list[l:r])
        l = r

    def compute_wx(w,x):
        if w.fit_intercept:
            return x.dot(w.coef_)+w.intercept_
        else:
            return x.dot(w.coef_)

    def get_half_d(data):
        y = data[1][0]
        x = np.array(data[1][1:])
        re = 0.25 * compute_wx(model_weights_v,x) - 0.5 * y
        return (data[0],re)

    def for_each_rdd(data_iter):
        socket = reConnect.Connect_gennerator.getClientConnect()
        print("进入send")
        for i in data_iter:
            print(i)
            data_s = pickle.dumps(i)
            if role == 'host':
                rec = socket.recv()
                socket.send(data_s)
            else:
                socket.send(data_s)
                rec = socket.recv()
        socket.close()

    for i in batch_index:
        batch_data = file_rdd.filter(lambda x:x[0] in i)
        #计算得到half g
        half_d = batch_data.map(get_half_d)
        #half_d可能要复用，所以cache在缓存中
        half_d.cache()
        en_half_d = half_d.map(lambda x:(x[0],pk.encrypt(x[1])))
        print("进入foreach")
        en_half_d.foreachPartition(for_each_rdd)
        socket = reConnect.Connect_gennerator.getClientConnect()
        socket.send(pickle.dumps(-1))
        socket.close()
