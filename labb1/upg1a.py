import numpy as np #hanterar många x-värden samtidigt
import matplotlib.pyplot as plt #ritar grafer

def f(x):
    return (8/3)*x - 3*x**2 + (1/3)*x**3 - (2/3)*np.sin(np.pi*x)

def main():
    x = np.linspace(0, 1, 1000) #undre gräns, övre gräns, antal punkter
    plt.plot(x, f(x)) #ritar grafen till funktionen
    plt.axhline(0) #ritar horisontell linje vid y=0
    plt.xlabel("x")  #markerar x-axeln som "x"
    plt.ylabel("f(x)") #markerar y-axeln som "f(x)"
    plt.show() #visar grafen

if __name__ == "__main__":
    main()