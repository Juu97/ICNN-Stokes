import pickle
import matplotlib.pyplot as plt
import numpy as np

nn1 = [1.2, 1.6, 2.0, 2.4, 2.8]
nn2 = [2.4, 2.8]
mu_inf = 0.001
mu_0 = 1
lambd = 100

# Load data
def carreau(x, n):
    return mu_inf + (mu_0 - mu_inf) * (1 + lambd * x ** 2) ** ((n - 2) / 2)

x = np.logspace(np.log10(1e-3), np.log10(70), 40)
colors = ['blue', 'red', 'green', 'red', 'purple']

fig, axs = plt.subplots(1, 2, figsize=(14, 7))

for ax, nn in zip(axs, [nn1, nn2]):
    color_index = 0
    for n in nn:
        with open(f'n{str(n)[0]}-{str(n)[2]}.pkl', 'rb') as f:
            icnn = pickle.load(f)

        sign = 1 if n <= 2 else -1

        ax.loglog(x, sign * icnn(x), label=f'n={n}', ls='--', color=colors[color_index])
        ax.scatter(x, carreau(x, n), color=colors[color_index], s=12.5, marker='x', alpha=0.5)

        color_index += 1

    ax.grid()
    ax.legend(fontsize=14)
    ax.set_xlabel('Shear rate', fontsize=14)
    ax.set_ylabel('Viscosity', fontsize=14)


plt.tight_layout()

plt.show()
