"""LÖSNING UPPG 1c LABB 1 """

import numpy as np

def g(x):
    """Funktion g(x)"""
    g = ((9/8) * x**2 - (1/8) * x**3 + (1/4) * np.sin(np.pi*x))
    return g

def main():
    x0 = 0.5 #Startgissning där 0 < x < 1
    start_iteration = 0
    max_iteration = 300 #Maxantal iterationer
    tolerans = 1E-10 #Toleransen
    differensvärde = 1 #Startvärde för differensen av absoluta felet
    x_tidigare = x0

    while start_iteration < max_iteration and differensvärde > tolerans:
        x_nytt = g(x_tidigare) #Beräknar nytt värde på x
        differensvärde = abs(x_nytt - x_tidigare) #differensvärdet = absoluta felet
        print(f" Iterationsnummer: {start_iteration} startvärde x: {x_tidigare} nytt värde x: {x_nytt} differensvärde: {differensvärde}")
        x_tidigare = x_nytt # Lagrar det nya värdet
        start_iteration += 1 #Itererar nästa steg

if __name__=="__main__":
    main()