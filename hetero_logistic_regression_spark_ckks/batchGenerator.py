#-*- codeing = utf-8 -*-
# @Time : 2022/7/5 15:08
# @Author : 夏冰雹
# @File : batchGenerator.py 
# @Software: PyCharm

import numpy as np

def batch_generator(all_data , batch_size, shuffle=True):
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



if __name__ =="__main__":
    # 输入x表示有23个样本，每个样本有两个特征
    # 输出y表示有23个标签，每个标签取值为0或1
    x = np.array([[i,i] for i in range(23)])
    y = np.random.random(size=[23, 1])
    print(y)
    # count = x.shape[0]
    #
    # batch_size = 5
    # epochs = 20
    # batch_num = count // batch_size
    #
    # batch_gen = batch_generator([x, y], batch_size,False)
    #
    # for i in range(epochs):
    #     print("##### epoch %s ##### " % i)
    #     for j in range(batch_num):
    #         batch_x, batch_y = next(batch_gen)
    #         print("-----epoch=%s, batch=%s-----" % (i, j))
    #         print(batch_x, batch_y)
