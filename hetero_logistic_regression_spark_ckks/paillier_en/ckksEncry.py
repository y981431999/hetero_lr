import tenseal as ts


class CkksKeypair(object):
    def __init__(self):
        pass

    @staticmethod
    def generate_keypair(l = 8192,bit_size = [60, 40, 40, 60],k = 40):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=l,
            coeff_mod_bit_sizes=bit_size
        )
        context.global_scale = 2 ** k
        context.generate_galois_keys()
        privatekey = CkksPrivateKey(context.secret_key())
        #context.make_context_public()
        publickey = CkksPublicKey(context)

        return publickey, privatekey


class CkksPublicKey(object):
    def __init__(self,context):
        self.context = context
        self.pks = context.serialize(save_secret_key=True)
    def encrypt(self, value):
        cipher_text = ts.ckks_vector(self.context,value)
        return cipher_text

class CkksPrivateKey(object):
    def __init__(self,peivatekey):
        self.privatekey = peivatekey
    def decrypt(self, encrypted_number):
        pliant_text = encrypted_number.decrypt(self.privatekey).tolist()
        return pliant_text


