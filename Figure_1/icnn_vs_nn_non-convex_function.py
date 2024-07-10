import pickle
import numpy as np
from matplotlib import pyplot as plt
from ICNN import ICNN


def function(x):
    return np.abs(x) + np.sin(x)


x = np.linspace(-10, 10, 100)

#
number_of_epochs = 3000
lr = 0.01

icnn = ICNN([1, 120, 56, 1], activation_function="elu")
nn = ICNN([1, 120, 56, 1], activation_function="elu")

icnn.convex_training(x, function(x), learning_rate=lr, epochs=number_of_epochs, epsilon=30, do_convex_training=True)
nn.convex_training(x, function(x), learning_rate=lr, epochs=number_of_epochs, epsilon=30, do_convex_training=False)

# # Plot ICNN model

plt.plot(x, function(x), label=f'Data', linewidth=3, alpha=0.25, color='blue')
plt.plot(x, nn(x), '--', color='black', label='NN', linewidth=1.5)
plt.plot(x, icnn(x), '--', color='red',label='ICNN', linewidth=1.5)
plt.xlabel('x', fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.grid()
plt.show()
