#-*- codeing = utf-8 -*-
# @Time : 2022/7/6 10:50
# @Author : 夏冰雹
# @File : test.py 
# @Software: PyCharm

import tenseal as ts
import numpy as np
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40,40, 60]
          )
context.generate_galois_keys()
sk = context.secret_key()
context.make_context_public()
context.global_scale = 2**40

p10 = [60, 66, 73, 81, 90]
p20 = [1,1,1,1,1]

data = np.array([[1,1,1,],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])

e0 = ts.ckks_tensor(context,p10)
k = np.array([61.00000000003265, 67.00000000102041, 73.99999999987962, 82.00000000034136, 90.99999999970936])
print(k.dot(data))
e1 = e0+p20
print(e1.decrypt(sk).tolist())
e2 = e1.dot(data)


eb = e2.decrypt(sk).tolist()
print(eb)
