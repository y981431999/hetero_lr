#-*- codeing = utf-8 -*-
# @Time : 2022/8/25 15:38
# @Author : 夏冰雹
# @File : mt2.py 
# @Software: PyCharm

import tenseal as ts
import numpy as np

context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = pow(2, 40)

# generated galois keys in order to do rotation on ciphertext vectors
context.generate_galois_keys()

# generate vectors
np.random.seed(1)
vectors = [np.random.randn(3).tolist() for i in range(5)]
print(vectors)

enc_vectors1 = [ts.ckks_vector(context, v) for v in vectors]
#enc_vectors2 = [ts.ckks_vector(context, v) for v in vectors]
enc_vectors_add = [e1 + e2 for e1, e2 in zip(enc_vectors1, vectors)]

result1 = ts.CKKSVector.pack_vectors(enc_vectors_add)
print(result1.decrypt())
result2 = [r.decrypt() for r in enc_vectors_add]
print(result2)