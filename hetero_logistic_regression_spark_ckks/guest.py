#-*- codeing = utf-8 -*-
# @Time : 2022/7/5 13:36
# @Author : 夏冰雹
# @File : client.py 
# @Software: PyCharm
import tenseal as ts
import pickle
import sys
import datetime
import numpy as np
import pdb
import reConnect
from hete_lr_base import hetero_lr_base
from optim.Weights import LinearModelWeights
from util import activation
from util.evaluation import evaluator,classification_eva
from pyspark.sql import SparkSession
from pyspark import SparkConf,SparkContext
import functools
import time
class Guest(hetero_lr_base):
    def __init__(self):
        super().__init__()
        self.role = "guest"
        self.socket = reConnect.Connect_gennerator.getClientConnect()
        self.train_path = hetero_lr_base.m["data"]["train"]["guest"]
        self.test_path = hetero_lr_base.m["data"]["test"]["guest"]
        self.threshold = hetero_lr_base.m["threshold"]
        self.spark_conf = SparkConf().setAppName("test").setMaster("local[*]")
        self.spark_sc = SparkContext(conf=self.spark_conf)

    def compute_half_d(self, data_instances,y):
        half_d =0.25* self.compute_wx(self.model_weights,data_instances) - 0.5*y
        return half_d
    def  compute_gradient_dis(self,data):
        def compute_wx(w,x):
            if w.fit_intercept:
                return x.dot(w.coef_) + w.intercept_
            else:
                return x.dot(w.coef_)
        def compute_half_d_dis(data):
            y = data[1][0]
            if y == 0:
                y = -1
            x = np.array(data[1][1:])
            re = 0.25 * compute_wx(model_weights_v, x) - 0.5 * y
            return (data[0], re)
        def send_foreach(data_iter):
            socket = reConnect.Connect_gennerator.getClientConnect(True)
            data_s = pickle.dumps(list(data_iter))
            socket.send(data_s)
            socket.recv()
            # for i in data_iter:
            #     data_s = pickle.dumps(i)
            #     socket.send(data_s)
            #     socket.recv()
            socket.close()
        def get_g(x):
            cp = x[1][-1]
            re = []
            for i in range(len(x[1][0])):
                re.append(x[1][0][i]*cp)
            re.append(cp)
            return (x[0],re)
        def get_g_reduce(x,y):
            re = np.array(x[1]) + np.array(y[1])
            return (x[0], re)

        data.cache()
        count = data.count()
        model_weights_b = self.spark_sc.broadcast(self.model_weights)
        model_weights_v = model_weights_b.value
        half_d = data.map(compute_half_d_dis)
        # half_d可能要复用，所以cache在缓存中
        half_d.cache()
        pl_gradient = None
        if self.encrypt_mode == "paillier" or self.encrypt_mode == "fake":
            pkb = self.spark_sc.broadcast(self.public_key)
            pk = pkb.value
            en_half_d = half_d.map(lambda x: (x[0], pk.encrypt(x[1])))
            en_half_d.foreachPartition(send_foreach)
            stop_socket = reConnect.Connect_gennerator.getClientConnect(True)
            stop_socket.send(pickle.dumps(-1))
            stop_socket.recv()
            stop_socket.close()
            self.send_and_receive(1)
            host_half_d = self.recv_dis()
            half_g = host_half_d.join(half_d).map(lambda x:(x[0],(x[1][0]+x[1][1])))
            gradient = data.map(lambda x:(x[0],x[1][1:])).join(half_g).map(get_g).reduce(get_g_reduce)[1]
            pl_gradient = self.get_pl_gradient_dis(gradient, count)
        elif self.encrypt_mode == "ckks":
            def get_g_ckks(x):
                opk = ts.context_from(opk_bin)
                gd = ts.ckks_vector_from(opk, half_d_bin)
                re = gd.dot(x[1]).serialize()
                return (x[0],re)

            pkb = self.spark_sc.broadcast(self.public_key.pks)
            pk_bin = pkb.value
            pk = ts.context_from(pk_bin)


            opkb = self.spark_sc.broadcast(self.other_pubk.serialize())
            opk_bin = opkb.value

            forwards = half_d.sortByKey().map(lambda x: x[1]).collect()
            en_half_d = self.public_key.encrypt(forwards)
            host_half_d = self.send_and_receive(en_half_d.serialize())
            host_half_d = ts.ckks_vector_from(self.other_pubk, host_half_d)
            half_d = host_half_d + forwards
            host_half_d_b = self.spark_sc.broadcast(half_d.serialize())
            half_d_bin = host_half_d_b.value
            transpose_data = self.get_transpose_rdd(data)
            gradient_bin = transpose_data.map(get_g_ckks).sortByKey().map(lambda x: x[1]).collect()
            gradient = []
            for i in gradient_bin:
                gradient.append(ts.ckks_vector_from(self.other_pubk, i))
            iter_ = [1]*count
            gradient.append(half_d.dot(iter_))
            pl_gradient = self.get_pl_gradient_dis_ckks(gradient, count)
        else:
            raise Exception(f"{self.encrypt_mode} not support,just support ckks,paillier and fake.Please change it and restart")
        return pl_gradient

    def  compute_gradient(self,batch_x,batch_y):
        half_d = self.compute_half_d(batch_x, batch_y)

        """
        2.加密过程，分布式待优化
        """

        ed = self.public_key.encrypt_list(half_d)
        host_half_d = self.send_and_receive(ed)

        """
        3.加法过程，分布式待优化
        """
        half_g = host_half_d + half_d

        """
        dot过程，分布式待优化
        """
        gradient = half_g.dot(batch_x)
        intercept_ = half_g.dot([1] * len(batch_x))
        gradient = np.append(gradient, intercept_)
        pl_gradient = self.get_pl_gradient(gradient, batch_x)
        return pl_gradient

    def compute_gradient_ckks(self, batch_x, batch_y):
        half_d = self.compute_half_d(batch_x, batch_y)

        """
        2.加密过程，分布式待优化
        """

        ed = self.public_key.encrypt(half_d)
        host_half_d = self.send_and_receive(ed.serialize())
        host_half_d = ts.ckks_vector_from(self.other_pubk,host_half_d)

        """
        3.加法过程，分布式待优化
        """
        half_g = host_half_d + half_d

        """
        dot过程，分布式待优化Z
        """
        #gradient = half_g.dot(batch_x)
        gradient = self.get_cp_gradient(half_g,batch_x)
        intercept_ = half_g.dot([1] * len(batch_x))
        gradient = np.append(gradient, intercept_)
        pl_gradient = self.get_pl_gradient_ckks(gradient, batch_x)
        return pl_gradient

    def predict(self, data_instances,true_y):
        # data_features = self.transform(data_instances)
        pred_prob = self.compute_wx(self.model_weights, data_instances)
        host_probs = self.send_and_receive(0)
        pred_prob += host_probs
        pred_prob = list(map(lambda x: activation.sigmoid(x), pred_prob))
        threshold = self.get_threshold(true_y,pred_prob)
        predict_result = self.predict_score_to_output(pred_prob, classes=[-1, 1], threshold=threshold)
        return predict_result,pred_prob,threshold

    def get_threshold(self,true_y,pred_prob):
        fpr, tpr, thresholds, ks = evaluator.getKS(true_y, pred_prob)
        max_ks = 0
        re = self.threshold
        for i in range(len(thresholds)):
            if tpr[i] - fpr[i]>max_ks:
                max_ks = tpr[i] - fpr[i]
                re = thresholds[i]
        return re

    def predict_score_to_output(self, pred_prob, classes, threshold):
        class_neg,class_pos = classes[0],classes[1]
        pred_lable = list(map((lambda x: class_neg if x<threshold else class_pos),pred_prob))
        return pred_lable

    def fit(self):
        if self.back_end=="standalone":
            self.fit_standalone()
        elif self.back_end == "spark":
            self.fit_dis()
        else:
            raise Exception(f"don't support {self.back_end} currently! please choose 'standalone' or 'spark' as back end.")
    def fit_standalone(self):
        start_time = time.time()
        self.get_key()
        #datainstance:np.array()
        datainstance = self.readTrainData(self.train_path)
        self.model_weights = LinearModelWeights(np.append(np.array(([0]*len(self.header))),0),True)
        # x = np.array([[i, i] for i in range(23)])
        # y = np.random.randint(2, size=[23, 1])
        count = datainstance.shape[0]
        batch_num = count // self.batch_size + 1
        batch_gen = self.batch_generator([datainstance,self.lable], self.batch_size, False)
        for i in range(self.epochs):
            print("##### epoch %s ##### " % i)
            for j in range(batch_num):
                batch_x, batch_y = next(batch_gen)
                print("-----epoch=%s, batch=%s-----" % (i, j))
                if self.encrypt_mode == "ckks":
                    gradient = self.compute_gradient_ckks(batch_x, batch_y)
                else:
                    gradient = self.compute_gradient(batch_x,batch_y)
                if self.optimizer is not None:
                    gradient = self.optimizer.add_regular_to_grad(gradient, self.model_weights)
                delta_grad = self.optimizer.apply_gradients(gradient)
                self.optimizer.set_iters(self.optimizer.iters+1)
                self.model_weights = self.optimizer.update_model(self.model_weights, delta_grad)
        predict_data, predict_data_header, true_lable = self.readData(self.test_path)
        predict_result,pred_prob,threshold = self.predict(predict_data,true_lable)
        evaluation_result = classification_eva.getResult(self.lable,predict_result,pred_prob)
        end_time = time.time()
        with open("result.txt",'a') as f:
            f.write(f"耗时：{end_time-start_time}")
            # f.write(f"model_weights:{self.model_weights.unboxed}")
            # f.write("\n\n=============================================\n\n")
            # f.write(f"predict_result:{predict_result}")
            # f.write("\n\n=============================================\n\n")
            # f.write(f"pred_prob:{pred_prob}")
            # f.write("\n\n=============================================\n\n")
            # f.write(f"threshold:{threshold}")
            # f.write("\n\n=============================================\n\n")

            for i in evaluation_result:
                if i.split(":")[0] == "KS":
                    continue
                f.write(i)
                f.write("\n")
            f.write(f"batch:{self.batch_size}\n")
            f.write(f"learning_rate:{self.learning_rate}\n")
            f.write("\n\n=============================================\n\n")


    def fit_dis(self):
        start_time = time.time()
        self.get_key()
        # datainstance:spark.rdd
        datainstance = self.readDataDis(self.train_path)
        self.model_weights = LinearModelWeights(np.append(np.array(([0] * len(self.header))), 0), True)
        count = datainstance.count()
        id_list = datainstance.map(lambda x: x[0]).collect()
        batch_list = self.get_batch_list(id_list,self.batch_size)
        for i in range(self.epochs):
            print("##### epoch %s ##### " % i)
            for j in range(len(batch_list)):
                batch_data = datainstance.filter(lambda x:x[0] in batch_list[j])
                print("-----epoch=%s, batch=%s-----" % (i, j))
                gradient = self.compute_gradient_dis(batch_data)
                # if self.encrypt_mode == "ckks":
                #     gradient = self.compute_gradient_ckks(batch_x, batch_y)
                # else:
                #     gradient = self.compute_gradient(batch_x, batch_y)
                if self.optimizer is not None:
                    gradient = self.optimizer.add_regular_to_grad(gradient, self.model_weights)
                delta_grad = self.optimizer.apply_gradients(gradient)
                self.optimizer.set_iters(self.optimizer.iters + 1)
                self.model_weights = self.optimizer.update_model(self.model_weights, delta_grad)
        predict_data, predict_data_header, true_lable = self.readData(self.test_path)
        predict_result, pred_prob, threshold = self.predict(predict_data, true_lable)
        evaluation_result = classification_eva.getResult(true_lable, predict_result, pred_prob)
        end_time = time.time()
        current_time = datetime.datetime.now()
        print(current_time)
        with open("result.txt", 'a') as f:
            f.write(f"time:{current_time}")
            f.write(f"耗时：{end_time - start_time}")
            # f.write(f"model_weights:{self.model_weights.unboxed}")
            # f.write("\n\n=============================================\n\n")
            # f.write(f"predict_result:{predict_result}")
            # f.write("\n\n=============================================\n\n")
            # f.write(f"pred_prob:{pred_prob}")
            # f.write("\n\n=============================================\n\n")
            # f.write(f"threshold:{threshold}")
            # f.write("\n\n=============================================\n\n")

            for i in evaluation_result:
                if i.split(":")[0] == "KS":
                    continue
                f.write(i)
                f.write("\n")
            f.write(f"batch:{self.batch_size}\n")
            f.write(f"learning_rate:{self.learning_rate}\n")
            f.write("\n\n=============================================\n\n")
if __name__ == "__main__":
    g = Guest()
    g.fit()

