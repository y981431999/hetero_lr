#-*- codeing = utf-8 -*-
# @Time : 2022/8/11 11:59
# @Author : 夏冰雹
# @File : sizetest.py 
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
            coeff_mod_bit_sizes=[60,30,30,60]
          )

context.global_scale = 2**30
cpk = context.public_key()
csk = context.secret_key()

data = np.random.rand(569)
with open("o_data.d",'wb') as f:
    pickle.dump(data,f)

p_data = pk.encrypt_list(data)
plook = p_data[0]
print(sys.getsizeof(pickle.dumps(plook)))

with open("p_data.d",'wb') as f:
    pickle.dump(p_data,f)
c_data = ts.ckks_vector(context,data)
print(sys.getsizeof(c_data))
data_c = c_data.serialize()
print(sys.getsizeof(data_c))
with open("c_data.d",'wb') as f:
    pickle.dump(data_c,f)