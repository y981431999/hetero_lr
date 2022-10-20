#-*- codeing = utf-8 -*-
# @Time : 2022/7/6 12:12
# @Author : 夏冰雹
# @File : multest.py 
# @Software: PyCharm
import tenseal as ts
import numpy as np
from paillier_en.paillier import PaillierKeypair
import pickle
import sys
pk,sk = PaillierKeypair.generate_keypair(2048)
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60,40,40,60]
          )
context.global_scale = 2**40
context.generate_galois_keys()

a = [3,5]
b = np.array([[10,2,3,4,34],[3,2,5,6,56]])
ea = ts.ckks_vector(context,a)
eaa = ts.ckks_vector(context,[6,9])
def get_cp_gradient(half_g, batch_x):
    re = []
    trans_x = batch_x.transpose()
    for i in trans_x:
        re.append(half_g.dot(i))
    return re

re = get_cp_gradient(ea,b)
list_cre1 = ts.CKKSVector.pack_vectors([ea,eaa])
list_cre2 = ts.CKKSVector.pack_vectors(re)


pre = []
for i in re:
    pre.append(i.decrypt())
print(pre)
print(list_cre1.decrypt())
print(list_cre2.decrypt())


