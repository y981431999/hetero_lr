class FakeKeypair(object):
    def __init__(self):
        pass

    @staticmethod
    def generate_keypair():
        public_key = FakePublicKey()
        private_key = FakePrivateKey()

        return public_key, private_key


class FakePublicKey(object):
    def encrypt(self, value):
        return value
    def encrypt_list(self,value):
        return value

class FakePrivateKey(object):
    def __init__(self):
        pass
    def decrypt(self, encrypted_number):
        return encrypted_number

    def decrypt_list(self, encrypted_number):
        return encrypted_number

