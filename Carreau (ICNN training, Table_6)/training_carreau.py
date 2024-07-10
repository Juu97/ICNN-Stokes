import pickle
import numpy as np
from matplotlib import pyplot as plt
from ICNN import ICNN

nn = [1.2, 1.6, 2.0, 2.4, 2.8]

mu_inf = 0.001
mu_0 = 1
lambd = 100


def carreau(x, n):
    return mu_inf + (mu_0 - mu_inf) * (1 + lambd * x ** 2) ** ((n - 2) / 2)


x = np.logspace(np.log10(1e-3), np.log10(1e2), 200)

#
number_of_epochs = 50000
lr = 0.01

for n in nn:
    icnn = ICNN([1, 120, 56, 1], activation_function="elu")
    sign = 1 if n <= 2 else -1
    icnn.convex_training(x, sign * carreau(x, n), learning_rate=lr, epochs=number_of_epochs, epsilon=30)

    # # Plot ICNN model
    plt.plot(x, sign * icnn(x), label='ICNN')
    plt.plot(x, carreau(x, n), label=f'Carreau {n}')
    plt.legend()
    plt.show()

    with open(f'n{str(n)[0]}-{str(n)[2]}.pkl', 'wb') as f:
        pickle.dump(icnn, f)
