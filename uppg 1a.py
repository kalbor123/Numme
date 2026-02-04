"""LÖSNING UPPG 1 LABB 1 """

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (8/3)*x - 3*x**2 + (1/3)*x**3 - (2/3)*np.sin(np.pi*x)


def main():
    x = np.linspace(0, 1, 1000)
    plt.plot(x, f(x))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0)
    plt.show()

if __name__=="__main__":
    main()