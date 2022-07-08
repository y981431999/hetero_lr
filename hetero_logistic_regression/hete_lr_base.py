#-*- codeing = utf-8 -*-
# @Time : 2022/7/5 14:31
# @Author : 夏冰雹
# @File : hete_lr_base.py 
# @Software: PyCharm
import json
import numpy as np
import pickle
from paillier_en.paillier import PaillierKeypair
from paillier_en.fakeEncry import FakeKeypair
from optim.optimizer import _RMSPropOptimizer
class hetero_lr_base:
    path = 'param.json'
    f = open(path, 'r', encoding='utf-8')
    m = json.load(f)
    def __init__(self):
        self.public_key = None
        self.privite_key = None
        self.socket = None
        self.other_pubk = None
        self.role = None
        self.header = None
        self.batch_size = hetero_lr_base.m["batch_size"]
        self.learning_rate = hetero_lr_base.m["learning_rate"]
        self.alpha = hetero_lr_base.m["alpha"]
        self.epochs = hetero_lr_base.m["epochs"]
        self.n_length = hetero_lr_base.m["n_length"]
        self.encrypt_mode = hetero_lr_base.m["encrypt_mode"]
        self.model_weights = None
        # learning_rate, alpha, penalty, decay, decay_sqrt
        self.penalty = hetero_lr_base.m["penalty"]
        self.decay = hetero_lr_base.m["decay"]
        self.decay_sqrt = hetero_lr_base.m["decay_sqrt"]
        self.optimizer = _RMSPropOptimizer(self.learning_rate,self.alpha,self.penalty,self.decay,self.decay_sqrt)
        self.lable = None

    def send_and_receive(self,value):
        b =pickle.dumps(value)
        if self.role == 'host':
            rec = self.socket.recv()
            self.socket.send(b)
        else:
            self.socket.send(b)
            rec = self.socket.recv()
        rec = pickle.loads(rec)
        return rec

    def compute_wx(self,w,x):
        if w.fit_intercept:
            return x.dot(w.coef_)+w.intercept_
        else:
            return x.dot(w.coef_)

    def readData(self,path,delimiter =','):
        data = np.loadtxt(path,str,delimiter = delimiter)
        header = data[0]
        re_data = []
        this_row = []
        lable = None
        for i in data[1:]:
            this_row.clear()
            for j in range(len(i)):
                this_row.append(float(i[j]))
            re_data.append(this_row.copy())
        datainstance = np.array(re_data)
        if self.role == "guest":
            lable_index = list(header).index('y')
            lable = datainstance[:, lable_index]
            for i in range(len(lable)):
                if lable[i] == 0:
                    lable[i] = -1
            datainstance = np.delete(datainstance,lable_index, axis = 1)
            header = np.delete(header,lable_index)
        id_index = list(header).index('id')
        header = np.delete(header,id_index)
        datainstance = np.delete(datainstance, id_index, axis=1)
        return datainstance,header,lable

    def readTrainData(self,path,delimiter=','):
        datainstance,self.header,self.lable = self.readData(path,delimiter)
        return datainstance

    def readPredictData(self,path,delimiter =','):
        predict_data,predict_header,predict_lable = self.readData(path,delimiter)
        return predict_data,predict_header,predict_lable



    def get_key(self):
        if self.encrypt_mode == "paillier":
            self.public_key,self.privite_key = PaillierKeypair.generate_keypair(self.n_length)
            self.other_pubk = self.send_and_receive(self.public_key)
        elif self.encrypt_mode == "ckks":
            pass
        elif self.encrypt_mode == "fake":
            self.public_key,self.privite_key = FakeKeypair.generate_keypair()
            self.other_pubk = self.send_and_receive(self.public_key)

    def get_pl_gradient(self,gradient,batch_x):
        if self.role == "guest":
            mask = np.random.random(size=[len(self.header) + 1])
        else:
            mask = np.random.random(size=[len(self.header)])
        mask_gradient = gradient + mask
        need_de_mask = self.send_and_receive(mask_gradient)
        de_mask = self.privite_key.decrypt_list(need_de_mask)
        pl_mask_gradient = self.send_and_receive(de_mask)
        pl_gradient = pl_mask_gradient - mask
        pl_gradient /= len(batch_x)
        return pl_gradient

    def batch_generator(self,all_data, batch_size, shuffle=True):
        """
        :param all_data : all_data整个数据集，包含输入和输出标签
        :param batch_size: batch_size表示每个batch的大小
        :param shuffle: 是否打乱顺序
        :return:
        """
        # 输入all_datas的每一项必须是numpy数组，保证后面能按p所示取值
        all_data = [np.array(d) for d in all_data]
        # 获取样本大小
        data_size = all_data[0].shape[0]
        print("data_size: ", data_size)
        if shuffle:
            # 随机生成打乱的索引
            p = np.random.permutation(data_size)
            # 重新组织数据
            all_data = [d[p] for d in all_data]
        batch_count = 0
        while True:
            # 数据一轮循环(epoch)完成，打乱一次顺序
            if batch_count * batch_size + batch_size > data_size:
                batch_count = 0
                if shuffle:
                    p = np.random.permutation(data_size)
                    all_data = [d[p] for d in all_data]
            start = batch_count * batch_size
            end = start + batch_size
            batch_count += 1
            yield [d[start: end] for d in all_data]