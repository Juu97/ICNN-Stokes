import pickle

import numpy as np
from matplotlib import pyplot as plt

from ICNN import ICNN

n = 1.6

mu_inf = 0.001
mu_0 = 1
lambd = 100


# Load data
def carreau(x):
    return mu_inf + (mu_0 - mu_inf) * (1 + lambd * x ** 2) ** ((n - 2) / 2)


x = np.logspace(np.log10(1e-3), np.log10(1e2), 200)

#
training_epochs = [800, 1600, 3200, 50000]
lr = 0.01

for epochs in training_epochs:
    icnn = ICNN([1, 120, 56, 1], activation_function="elu")
    sign = 1 if n <= 2 else -1
    icnn.convex_training(x, sign * carreau(x), learning_rate=lr, epochs=epochs, epsilon=30)
    print(epochs, icnn.training_val_loss)

    val_loss = str(round(icnn.training_val_loss, 8))
    with open(f'n{str(n)[0]}-{str(n)[2]}___{epochs}___{val_loss[0]}-{val_loss[2:]}.pkl', 'wb') as f:
        pickle.dump(icnn, f)
