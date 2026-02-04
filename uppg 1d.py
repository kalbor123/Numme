"""LÖSNING UPPG 1d LABB 1 """

import numpy as np

def g(x):
    """Funktion g(x)"""
    g = ((9/8) * x**2 - (1/8) * x**3 + (1/4) * np.sin(np.pi*x))
    return g

def f(x):
    """Funktion f(x)"""
    f = (8/3) * x - 3*x**2 + (1/3) * x**3 - (2/3) * np.sin(np.pi*x)
    return f

def derivata_f(x):
    """Derivatan till f(x)"""
    derivata_f = (8/3) - 6*x + x**2 - ((2*np.pi)/3) * np.cos(np.pi*x)
    return derivata_f


def main():
    x0 = 0.5 #Startgissning för metod med fixpunktiteration där 0 < x < 1
    iteration = 0
    max_iteration = 300 #Säkrar maxantal iterationer
    tolerans = 1E-10 #Toleransen
    differensvärde = 1 #Startvärde för differensen av absoluta felet
    x_tidigare = x0

    while iteration < max_iteration and differensvärde > tolerans:
        x_nytt = g(x_tidigare) #Beräknar nytt värde på x
        differensvärde = abs(x_nytt - x_tidigare) #differensvärdet = absoluta felet
        print(f" Iterationsnummer: {iteration} startvärde x: {x_tidigare} nytt värde x: {x_nytt} differensvärde: {differensvärde}")
        x_tidigare = x_nytt # Lagrar det nya värdet
        iteration += 1 #Itererar nästa steg

    #Med Newtons metod
    x_tidigare = 0.4
    iteration = 0
    differensvärde = 1

    print("Newtons metod: ")

    while iteration < max_iteration and differensvärde > tolerans:
        x_nytt = x_tidigare - (f(x_tidigare)/derivata_f(x_tidigare)) #Newtons metod
        differensvärde = abs(x_nytt - x_tidigare) #absoluta felet
        print(f" Iterationsnummer: {iteration} startvärde x: {x_tidigare} nytt värde x: {x_nytt} differensvärde: {differensvärde}")
        x_tidigare = x_nytt #Lagrar nyligen beräknat värde x
        iteration +=1 #itererar nästa steg

if __name__=="__main__":
    main()