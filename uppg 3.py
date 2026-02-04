import numpy as np
import matplotlib.pyplot as plt

t = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
y = np.array([12.00, 15.10, 19.01, 23.92, 30.11, 37.90, 47.70, 60.03, 75.56 ])


def trapetsregeln(f, a, b, N):
    """Definition trapetsregeln"""
    h = (b-a)/N
    x = np.linspace(a,b, N+1)
    y = f(x)
    #Beräknar värde med trapetsregeln
    I_T = h*(y[0] + 2*np.sum(y[1:-1]) + y[-1])/2
    return I_T

def given_funktion(x):
    """Returnerar funktion från uppgift a"""
    return x**3 * np.exp(x)

def uppg_a():
    """Returnerar approximativt värde på beräknad integral"""
    a = 0
    b = 2
    N = 10000
    approx = trapetsregeln(given_funktion,a, b, N)
    return approx

def uppgift_b():
    """Gör konvergensstudie, beräknar felskattning från exakt värde"""
    a = 0
    b = 2
    N_lista = [10, 20, 40, 80, 160, 320 , 640, 1280]
    I_exakt = 6 + 2 * np.exp(2)
    föregående_felskattning = 0
    for N in N_lista:
        approximering = trapetsregeln(given_funktion, a,b, N)
        felskattning = abs(I_exakt-approximering)
        print (f"N-värde: {N:<15} {felskattning:<15.6f}  {approximering:.6f}")
        if föregående_felskattning != 0:
            kvot = föregående_felskattning/ felskattning
            print(f"Kvot:{kvot:.6f}")
            noggrannhetsordning = np.log2(kvot)
            print(f"Nogrannhetsordning:{noggrannhetsordning:.6f}")
        föregående_felskattning = felskattning

def uppg_c(y):
    """Skattar värdet med integral 3 genom trapetsregeln"""
    h = 1
    skattat_värde = h*(y[0] + 2*np.sum(y[1:-1]) + y[-1])/2
    return skattat_värde


def uppg_d(y):
    """Konvergensstudie med olika värden på h"""
    h_lista = [1,2,4,8]
    föregående_skattevärde = 0
    föregående_felskattning = 0
    for h in h_lista:
        f_h = y[::h]
        skattat_värde = h*(f_h[0] + 2*np.sum(f_h[1:-1]) + f_h[-1])/2
        print(f"Antal steg: {h}, skattat värde: {skattat_värde:.6f}")
        if föregående_skattevärde != 0:
            felskattning = abs(föregående_skattevärde - skattat_värde)
            print(f"Felskattning: {felskattning:.6f}")
            if föregående_felskattning !=0:
                kvot = föregående_felskattning/felskattning
                noggrannhetsordning = np.log2(kvot)
                print(f"Kvot {kvot:.6f} Nogrannhetsordning:{noggrannhetsordning:.6f}")
            föregående_felskattning = felskattning
        föregående_skattevärde = skattat_värde

def richardssons_extrapolation():
    """Richardsons extrapolation av två skattningar"""
    I_T1 = 277.55
    I_T2 = 281.20
    q = 2 
    p=2
    I = I_T1 + ((I_T1 - I_T2))/(q**p - 1)
    return I

def simpsons_regel(h):
    """Definierar simpsons regel"""
    I_S = h*(y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])/3
    return I_S

def approx_simpsons():
    """Approximerar integral 3 med simpsons regel """
    h = 1
    approx = simpsons_regel(h)
    return approx

def main():

    svar_a = uppg_a()
    print(f"Approximerat värde uppgift a:\n{svar_a}")

    print("Uppgift b:")
    uppgift_b()

    print("Uppgift c:")
    energimängd = uppg_c(y)
    print(f"Skattad energimängd för åren 2014-2022:\n{energimängd}kW")

    print("Uppgift d")
    uppg_d(y)

    print("Uppgift e:")
    print("Med Richardsons extrapolation ges skattat värde:")
    richardsons = richardssons_extrapolation()
    print(richardsons)
    print("Med Simpsons regel ges:")
    approx_värde = approx_simpsons()
    print(approx_värde)

if __name__=='__main__':
    main()
