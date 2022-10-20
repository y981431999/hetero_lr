#-*- codeing = utf-8 -*-
# @Time : 2022/7/6 10:50
# @Author : 夏冰雹
# @File : test.py 
# @Software: PyCharm
import pickle
import tenseal as ts
import numpy as np
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40,40, 60]
          )
context.generate_galois_keys()
sk = context.secret_key()
pk = context
context.make_context_public()
context.global_scale = 2**40
context.serialize()
guest_halfd = [0.7]

host_halfd = np.array([[1,2,3,4],[2,3,4,5]])
cipher_g = ts.ckks_tensor(pk,guest_halfd)
cs = cipher_g.serialize()
cs2 = ts.ckks_tensor_from(context,cs).decrypt(sk).tolist()
print(cs2)




# re = cipher_g.dot(host_halfd)
re = cipher_g+[0.2]
print(re.decrypt(sk).tolist())




# p10 = [60, 66, 73, 81, 90]
# p20 = [1,1,1,1,1]
#
# data = np.array([[1,1,1,],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])
#
# e0 = ts.ckks_tensor(context,p10)
# e6 = e0.serialize()
# k = np.array([61.00000000003265, 67.00000000102041, 73.99999999987962, 82.00000000034136, 90.99999999970936])
# print(k.dot(data))
# e1 = e0+p20
# print(e1.decrypt(sk).tolist())
# e2 = e1.dot(data)
#
#
# eb = e2.decrypt(sk).tolist()
# print(eb)
