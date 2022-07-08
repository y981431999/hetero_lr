#-*- codeing = utf-8 -*-
# @Time : 2022/7/5 13:36
# @Author : 夏冰雹
# @File : client.py 
# @Software: PyCharm
import numpy as np
import reConnect
from hete_lr_base import hetero_lr_base
from optim.Weights import LinearModelWeights
from util import activation
from util.evaluation import evaluator,classification_eva
import time
class Guest(hetero_lr_base):
    def __init__(self):
        super().__init__()
        self.role = "guest"
        self.socket = reConnect.Connect_gennerator.getClientConnect()
        self.train_path = hetero_lr_base.m["data"]["train"]["guest"]
        self.test_path = hetero_lr_base.m["data"]["test"]["guest"]
        self.threshold = hetero_lr_base.m["threshold"]
    def compute_half_d(self, data_instances,y):

        """
        1.计算half_d
        分布式待优化
        """

        half_d =0.25* self.compute_wx(self.model_weights,data_instances) - 0.5*y
         # half_d = data_instances.mapValues(
         #     lambda v: 0.25 * (vec_dot(v.features, w.coef_) + w.intercept_) - 0.5 * v.label)
        return half_d

    def compute_gradient(self,batch_x,batch_y,data_instance):
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
                gradient = self.compute_gradient(batch_x,batch_y,datainstance)
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
if __name__ == "__main__":
    g = Guest()
    g.fit()

