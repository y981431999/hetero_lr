# hetero_lr
decentralized hetero logistic regression by random mask
the parameters of the model in param.json

the code in hetero_logistic_regression/.

guest's data: Xb and y
host's data: only Xa
guest: hetero_logistic_regression/guest.py
host:hetero_logistic_regression/host.py

Guest and host communicate using RMQ.

the result of the evaluation in hetero_logistic_regression/result.txt

encrypt_mode only support "fake" and "pailllier". In fake mode, data interaction is not encrypted, and it will take less time to train the model. In paillier mode,data is encrypted by paillier homomorphic encryption algorithm,and in this project the code of paillier is from FATA of WeBank.

please set hetero_logistic_regression/ as a sorceroot.
