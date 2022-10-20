#-*- codeing = utf-8 -*-
# @Time : 2022/7/5 13:36
# @Author : 夏冰雹
# @File : server.py 
# @Software: PyCharm
import tenseal as ts
import numpy as np
import pdb
import reConnect
import functools
import pickle
from pyspark.sql import SparkSession
from pyspark import SparkConf,SparkContext
from hete_lr_base import hetero_lr_base
from optim.Weights import LinearModelWeights
class Host(hetero_lr_base):
    def __init__(self):
        super().__init__()
        self.role = "host"
        self.socket = reConnect.Connect_gennerator.getServerConnect()
        self.train_path = hetero_lr_base.m["data"]["train"]["host"]
        self.test_path = hetero_lr_base.m["data"]["test"]["host"]
        self.spark_conf = SparkConf().setAppName("test").setMaster("local[*]")
        self.spark_sc = SparkContext(conf=self.spark_conf)

    # def compute_compute_forwards_dis(self, data,model_weights):
    #     y = data[1][0]
    #     x = np.array(data[1][1:])
    #     re = 0.25 * self.compute_wx(model_weights, x)
    #     return (data[0], re)

    def compute_forwards(self, data_instances):
        """
        forwards = 1/4 * wx
        """
        # wx = data_instances.mapValues(lambda v: vec_dot(v.features, model_weights.coef_) + model_weights.intercept_)
        forwards = 0.25 * self.compute_wx(self.model_weights,data_instances)
        return forwards

    def compute_gradient(self,batch_x):
        half_d = self.compute_forwards(batch_x)
        ed = self.public_key.encrypt_list(half_d)
        guest_half_d = self.send_and_receive(ed)
        half_g = half_d+guest_half_d
        gradient = half_g.dot(batch_x)
        pl_gradient = self.get_pl_gradient(gradient,batch_x)
        return pl_gradient

    def compute_gradient_dis(self,data):
        def compute_wx(w,x):
            if w.fit_intercept:
                return x.dot(w.coef_) + w.intercept_
            else:
                return x.dot(w.coef_)

        def compute_forwards_dis(data):
            x = np.array(data[1])
            re = 0.25 * compute_wx(model_weights_v, x)
            return (data[0], re)

        def send_foreach(data_iter):
            socket = reConnect.Connect_gennerator.getClientConnect(True)
            data_s = pickle.dumps(list(data_iter))
            socket.send(data_s)
            socket.recv()
            socket.close()
        def get_g(x):
            cp = x[1][-1]
            re = []
            for i in range(len(x[1][0])):
                re.append(x[1][0][i]*cp)
            return (x[0],re)
        def get_g_reduce(x,y):
            re = np.array(x[1]) + np.array(y[1])
            return (x[0], re)

        count = data.count()
        model_weights_b = self.spark_sc.broadcast(self.model_weights)
        model_weights_v = model_weights_b.value
        pl_gradient = None
        data.cache()
        forwards = data.map(compute_forwards_dis)
        forwards.cache()
        if self.encrypt_mode=="paillier" or self.encrypt_mode =="fake":
            pkb = self.spark_sc.broadcast(self.public_key)
            pk = pkb.value
            en_half_d = forwards.map(lambda x: (x[0], pk.encrypt(x[1])))
            guest_half_d = self.recv_dis()
            self.send_and_receive(1)
            en_half_d.foreachPartition(send_foreach)
            stop_socket = reConnect.Connect_gennerator.getClientConnect(True)
            stop_socket.send(pickle.dumps(-1))
            stop_socket.recv()
            stop_socket.close()
            half_g = forwards.join(guest_half_d).map(lambda x: (x[0], (x[1][0] + x[1][1])))
            gradient = data.join(half_g).map(get_g).reduce(get_g_reduce)[1]
            pl_gradient = self.get_pl_gradient_dis(gradient, count)
        elif self.encrypt_mode=="ckks":
            def get_g_ckks(x):
                opk = ts.context_from(opk_bin)
                gd = ts.ckks_vector_from(opk, guest_half_d)
                #gd=np.array([-0.5,-0.5,-0.5,0.5])
                re = gd.dot(x[1]).serialize()
                return (x[0], re)
            pkb = self.spark_sc.broadcast(self.public_key.pks)
            pk_bin = pkb.value
            pk = ts.context_from(pk_bin)
            opkb = self.spark_sc.broadcast(self.other_pubk.serialize())
            opk_bin = opkb.value
            forwards = forwards.sortByKey().map(lambda x:x[1]).collect()
            en_half_d = self.public_key.encrypt(forwards)
            guest_half_g = self.send_and_receive(en_half_d.serialize())
            guest_half_g = ts.ckks_vector_from(self.other_pubk, guest_half_g)
            half_g = guest_half_g + forwards
            guest_half_d_b = self.spark_sc.broadcast(half_g.serialize())
            guest_half_d = guest_half_d_b.value
            transpose_data = self.get_transpose_rdd(data)
            gradient_bin = transpose_data.map(get_g_ckks).sortByKey().map(lambda x:x[1]).collect()
            gradient = []
            for i in gradient_bin:
                gradient.append(ts.ckks_vector_from(self.other_pubk, i))
            pl_gradient = self.get_pl_gradient_dis_ckks(gradient, count)
        else:
            raise Exception(f"{self.encrypt_mode} not support,just support ckks,paillier and fake.Please change it and restart")

        return pl_gradient

    def compute_gradient_ckks(self,batch_x):
        half_d = self.compute_forwards(batch_x)
        ed = self.public_key.encrypt(half_d)
        guest_half_d = self.send_and_receive(ed.serialize())
        guest_half_d = ts.ckks_vector_from(self.other_pubk, guest_half_d)
        half_g = guest_half_d + half_d
        #gradient = half_g.dot(batch_x)
        gradient = self.get_cp_gradient(half_g, batch_x)
        pl_gradient = self.get_pl_gradient_ckks(gradient,batch_x)
        return pl_gradient

    def predict(self,datainstance):
        pre_prob = self.compute_wx(self.model_weights,datainstance)
        self.send_and_receive(pre_prob)

    def fit(self):
        if self.back_end=="standalone":
            self.fit_standalone()
        elif self.back_end == "spark":
            self.fit_dis()
        else:
            raise Exception(f"don't support {self.back_end} currently! please choose 'standalone' or 'spark' as back end.")

    def fit_standalone(self):
        self.get_key()
        datainstance = self.readTrainData(self.train_path)
        self.model_weights = LinearModelWeights(np.array(([0] * len(self.header))),False)
        count = datainstance.shape[0]
        batch_num = count // self.batch_size +1
        batch_gen = self.batch_generator([datainstance], self.batch_size, False)
        for i in range(self.epochs):
            print("##### epoch %s ##### " % i)
            for j in range(batch_num):
                batch_x = next(batch_gen)[0]
                print("-----epoch=%s, batch=%s-----" % (i, j))
                if self.encrypt_mode == "ckks":
                    gradient = self.compute_gradient_ckks(batch_x)
                else:
                    gradient = self.compute_gradient(batch_x)
                if self.optimizer is not None:
                    gradient = self.optimizer.add_regular_to_grad(gradient, self.model_weights)
                delta_grad = self.optimizer.apply_gradients(gradient)
                self.optimizer.set_iters(self.optimizer.iters+1)
                self.model_weights = self.optimizer.update_model(self.model_weights, delta_grad)
        #预测过程
        predict_data,predict_data_header,true_lable = self.readData(self.test_path)
        self.predict(predict_data)

    def fit_dis(self):
        self.get_key()
        # datainstance:rdd()
        datainstance = self.readDataDis(self.train_path)
        self.model_weights = LinearModelWeights(np.array(([0] * len(self.header))),False)
        batch_list = self.get_batch_list_host()
        for i in range(self.epochs):
            print("##### epoch %s ##### " % i)
            for j in range(len(batch_list)):
                batch_x = datainstance.filter(lambda x:x[0] in batch_list[j])
                print("-----epoch=%s, batch=%s-----" % (i, j))
                gradient = self.compute_gradient_dis(batch_x)
                if self.optimizer is not None:
                    gradient = self.optimizer.add_regular_to_grad(gradient, self.model_weights)
                delta_grad = self.optimizer.apply_gradients(gradient)
                self.optimizer.set_iters(self.optimizer.iters+1)
                self.model_weights = self.optimizer.update_model(self.model_weights, delta_grad)
        #预测过程
        predict_data,predict_data_header,true_lable = self.readData(self.test_path)
        self.predict(predict_data)
if __name__ == "__main__":
    host = Host()
    host.fit()

