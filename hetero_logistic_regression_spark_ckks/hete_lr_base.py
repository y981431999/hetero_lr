#-*- codeing = utf-8 -*-
# @Time : 2022/7/5 14:31
# @Author : 夏冰雹
# @File : hete_lr_base.py 
# @Software: PyCharm
import json
import numpy as np
import sys
import pdb
import pickle
import reConnect
import tenseal as ts
from pyspark.sql import SparkSession
from pyspark import SparkConf,SparkContext
from paillier_en.paillier import PaillierKeypair
from paillier_en.fakeEncry import FakeKeypair
from paillier_en.ckksEncry import CkksKeypair
from optim.optimizer import _RMSPropOptimizer
SUM_SIZE = 0
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
        self.n_lengths = hetero_lr_base.m["n_lengths"]
        self.encrypt_mode = hetero_lr_base.m["encrypt_mode"]
        self.model_weights = None
        self.context = None
        self.penalty = hetero_lr_base.m["penalty"]
        self.decay = hetero_lr_base.m["decay"]
        self.decay_sqrt = hetero_lr_base.m["decay_sqrt"]
        self.back_end = hetero_lr_base.m["back_end"]
        self.spark_conf = None
        self.spark_sc = None
        self.optimizer = _RMSPropOptimizer(self.learning_rate,self.alpha,self.penalty,self.decay,self.decay_sqrt)
        self.lable = None
        self.header = None

    def send_and_receive(self,value):
        b =pickle.dumps(value)
        print(sys.getsizeof(b))
        if self.role == 'host':
            rec = self.socket.recv()
            self.socket.send(b)
        else:
            self.socket.send(b)
            rec = self.socket.recv()
        rec = pickle.loads(rec)
        return rec

    def recv_dis(self):
        socket = reConnect.Connect_gennerator.getServerConnect(True)
        guest_rdd = self.spark_sc.parallelize([], 4)
        data = pickle.dumps(0)
        while True:
            rec = socket.recv()
            socket.send(data)
            rec = pickle.loads(rec)
            if rec == -1:
                break
            else:
                rdd = self.spark_sc.parallelize(rec)
                guest_rdd = guest_rdd.union(rdd)
        socket.close()
        return guest_rdd

    def compute_wx(self,w,x):
        if w.fit_intercept:
            return x.dot(w.coef_)+w.intercept_
        else:
            return x.dot(w.coef_)

    def get_half_d_dis(self,data):
        def compute_wx(w, x):
            if w.fit_intercept:
                return x.dot(w.coef_) + w.intercept_
            else:
                return x.dot(w.coef_)
        y = data[1][0]
        x = np.array(data[1][1:])
        re = 0.25 * compute_wx(self.model_weights, x) - 0.5 * y
        return (data[0], re)

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

    def get_batch_list(self,id_list,batch_size):
        batch_index = []
        l = 0
        while True:
            if l >= len(id_list):
                break
            r = l + batch_size
            if r >= len(id_list):
                r = len(id_list)
            batch_index.append(id_list[l:r])
            l = r
        self.send_and_receive(batch_index)
        return batch_index

    def get_batch_list_host(self):
        batch_list = self.send_and_receive(0)
        return batch_list

    def readDataDis(self,path):
        file_rdd = self.spark_sc.textFile(path,4)
        header = file_rdd.first().split(',')
        file_rdd = file_rdd.filter(lambda x: x[0] != header[0][0])

        def pre_proc_data(data):
            d_l = data.strip().split(',')
            key = int(d_l[0])
            value = list(map(float, d_l[1:]))
            return (key, (value))
        file_rdd = file_rdd.map(pre_proc_data)
        if self.role == "guest":
            self.header = header[2:]
        else:
            self.header = header[1:]
        return file_rdd

    def readPredictData(self,path,delimiter =','):
        predict_data,predict_header,predict_lable = self.readData(path,delimiter)
        return predict_data,predict_header,predict_lable



    def get_key(self):
        if self.encrypt_mode == "paillier":
            self.public_key,self.privite_key = PaillierKeypair.generate_keypair(self.n_length)
            self.other_pubk = self.send_and_receive(self.public_key)
        elif self.encrypt_mode == "ckks":
            self.public_key, self.privite_key = CkksKeypair.generate_keypair(self.n_lengths[0],self.n_lengths[1],self.n_lengths[2])
            rec = self.send_and_receive(self.public_key.pks)
            self.other_pubk = ts.context_from(rec)
        elif self.encrypt_mode == "fake":
            self.public_key,self.privite_key = FakeKeypair.generate_keypair()
            self.other_pubk = self.send_and_receive(self.public_key)
        else:
            raise Exception(f"encrypt mode:{self.encrypt_mode} is not support!please choose paillier,ckks or fake.")

    def get_pl_gradient_ckks(self,gradient,batch_x):
        if self.role == "guest":
            mask_gradient = []
            mask = np.random.random(size=[len(self.header) + 1])
            for i in range(len(gradient)):
                mask_gradient.append((gradient[i]+mask[i]).serialize())
            # mask_gradient.append((gradient[0] + mask[:-1]).serialize())
            #mask_gradient.append((gradient[1] + mask[-1]).serialize())
            need_de_mask = self.send_and_receive(mask_gradient)
            de_mask = []
            for i in need_de_mask:
                de_mask.append(ts.ckks_vector_from(self.public_key.context,i).decrypt(self.privite_key.privatekey)[0])
        else:
            mask_gradient = []
            mask = np.random.random(size=[len(self.header)])
            for i in range(len(gradient)):
                mask_gradient.append((gradient[i]+mask[i]).serialize())
            need_de_mask = self.send_and_receive(mask_gradient)
            # de_mask = ts.ckks_tensor_from(self.public_key.context,need_de_mask[0]).decrypt(self.privite_key.privatekey).tolist()
            de_mask = []
            for i in need_de_mask:
                de_mask.append(ts.ckks_vector_from(self.public_key.context, i).decrypt(self.privite_key.privatekey)[0])
            # inter_ = ts.ckks_tensor_from(self.public_key.context,need_de_mask[1]).decrypt(self.privite_key.privatekey).tolist()
            # de_mask.append(inter_)
        #     pl_mask_gradient = self.send_and_receive(de_mask)
        #
        # need_de_mask = self.send_and_receive(mask_gradient)
        # de_mask = self.privite_key.decrypt_list(need_de_mask)
        pl_mask_gradient = self.send_and_receive(de_mask)
        pl_gradient = pl_mask_gradient - mask
        pl_gradient /= len(batch_x)
        return pl_gradient

    def get_transpose_rdd(self,file_rdd3):
        start_index = 0 if self.role=="host" else 1
        def add_index(data):
            result_list = []
            for i in range(start_index,len(data[1])):
                result_list.append((int(data[0]), i, data[1][i]))
            return result_list
        file_rdd4 = file_rdd3.map(add_index)
        file_rdd5 = file_rdd4.flatMap(lambda x: x)
        file_rdd6 = file_rdd5.groupBy(lambda x: x[1])
        def get_list(data):
            re = []
            l = list(data[1])
            l.sort(key=lambda x: x[0])
            for i in range(len(l)):
                re.append(float(l[i][2]))
            return (data[0], re)

        file_rdd7 = file_rdd6.map(get_list)
        file_rdd8 = file_rdd7.map(lambda x: (x[0], list(x[1])))
        return file_rdd8

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

    def get_pl_gradient_dis(self,gradient,count):
        if self.role == "guest":
            mask = np.random.random(size=[len(self.header) + 1])
        else:
            mask = np.random.random(size=[len(self.header)])
        mask_gradient = gradient + mask
        need_de_mask = self.send_and_receive(mask_gradient)
        de_mask = self.privite_key.decrypt_list(need_de_mask)
        pl_mask_gradient = self.send_and_receive(de_mask)
        pl_gradient = pl_mask_gradient - mask
        pl_gradient /= count
        return pl_gradient

    def get_pl_gradient_dis_ckks(self,gradient,count):
        if self.role == "guest":
            mask = np.random.random(size=[len(self.header) + 1])
        else:
            mask = np.random.random(size=[len(self.header)])
        mask_gradient = gradient + mask
        mask_gradient_list=[]
        for i in mask_gradient:
            mask_gradient_list.append(i.serialize())
        #mask_gradient = ts.CKKSVector.pack_vectors(mask_gradient).serialize()
        need_de_mask_bin = self.send_and_receive(mask_gradient_list)
        need_de_mask = []
        de_mask = []
        for i in need_de_mask_bin:
            de_mask.append(ts.ckks_vector_from(self.public_key.context,i).decrypt(self.privite_key.privatekey)[0])
        #de_mask = need_de_mask.decrypt(self.privite_key.privatekey)
        pl_mask_gradient = self.send_and_receive(de_mask)
        pl_gradient = pl_mask_gradient - mask
        pl_gradient /= count
        return pl_gradient

    def get_cp_gradient(self,half_g,batch_x):
        re = []
        trans_x = batch_x.transpose()
        for i in trans_x:
            re.append(half_g.dot(i))
        return re

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