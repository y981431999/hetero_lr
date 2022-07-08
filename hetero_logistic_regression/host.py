#-*- codeing = utf-8 -*-
# @Time : 2022/7/5 13:36
# @Author : 夏冰雹
# @File : server.py 
# @Software: PyCharm
import zmq
import time
import pickle

#-*- codeing = utf-8 -*-
# @Time : 2022/7/5 13:36
# @Author : 夏冰雹
# @File : client.py
# @Software: PyCharm
import numpy as np
import reConnect
from hete_lr_base import hetero_lr_base
from optim.Weights import LinearModelWeights
class Host(hetero_lr_base):
    def __init__(self):
        super().__init__()
        self.role = "host"
        self.socket = reConnect.Connect_gennerator.getServerConnect()
        self.train_path = hetero_lr_base.m["data"]["train"]["host"]
        self.test_path = hetero_lr_base.m["data"]["test"]["host"]

    def compute_forwards(self, data_instances):
        """
        forwards = 1/4 * wx
        """
        # wx = data_instances.mapValues(lambda v: vec_dot(v.features, model_weights.coef_) + model_weights.intercept_)
        forwards = 0.25 * self.compute_wx(self.model_weights,data_instances)
        return forwards

    def compute_gradient(self,batch_x,data_instance):
        half_d = self.compute_forwards(batch_x)
        ed = self.public_key.encrypt_list(half_d)
        guest_half_d = self.send_and_receive(ed)
        half_g = half_d+guest_half_d
        gradient = half_g.dot(batch_x)
        pl_gradient = self.get_pl_gradient(gradient,batch_x)
        return pl_gradient

    def predict(self,datainstance):
        pre_prob = self.compute_wx(self.model_weights,datainstance)
        self.send_and_receive(pre_prob)

    def fit(self):
        self.get_key()
        # datainstance:np.array()
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
                gradient = self.compute_gradient(batch_x,datainstance)
                if self.optimizer is not None:
                    gradient = self.optimizer.add_regular_to_grad(gradient, self.model_weights)
                delta_grad = self.optimizer.apply_gradients(gradient)
                self.optimizer.set_iters(self.optimizer.iters+1)
                self.model_weights = self.optimizer.update_model(self.model_weights, delta_grad)
        #预测过程
        predict_data,predict_data_header,true_lable = self.readData(self.test_path)
        self.predict(predict_data)
        print(self.model_weights.unboxed)

if __name__ == "__main__":
    host = Host()
    host.fit()

