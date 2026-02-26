import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


N = 4
L = 1
k = 2
h = L/N
TL = 2 #Randvillkor
TR = 2 #Randvillkor


def matris_A():
    return np.array([[-64, 32, 0],[32, -64, 32],[0, 32, -64]])

def q(x):
    return 50 * x **3 * np.log(x + 1)

def hl_uppg_a(TL, TR):
    """Högerledet för uppgift a"""
    b = np.array([q(0.25) - 32 *TL, q(0.5), q(0.75) - 32 *TR])
    return b


def generell_matris_A(N, k, h):
    """Generell matris för olika N"""
    n = N-1
    koefficient = k / h**2

    huvud_diagonal = -2 * koefficient * np.ones(n)
    sido_diagonal = koefficient * np.ones(n - 1)
    #Diagonaliserar i matris A
    A = np.diag(huvud_diagonal,0) + np.diag(sido_diagonal, -1) + np.diag(sido_diagonal, 1)
    return A


def hl_b(q, k, TL, TR, N, h, L):
    """Högerledet i uppgift b"""
    n = N - 1
    koefficient = k / h**2
    #De inre punkterna
    x_punkter = np.linspace(h,L-h,n) #Start en steglängd, stop på N -1 steg och n punkter

    b = q(x_punkter)
    b[0] = b[0] - (koefficient * TL) #Subtraktion från första värde q(x1)
    b[-1] = b[-1] - (koefficient * TR) #Subtraktion från sista värde q(xn)
    return b


def diskretisering_temperatur(N, q, k, TL, TR):
    """Returnerar gles matris"""
    L = 1
    h = L / N
    n = N - 1
    koefficient = k / h**2
    A_matris = generell_matris_A(N, k, h)
    A_gles = sparse.csr_matrix(A_matris)
    x_punkter = np.linspace(h,L-h,n)

    HL = q(x_punkter)
    HL[0] = HL[0] - (koefficient * TL) #Subtraktion från första värde q(x1)
    HL[-1] = HL[-1] - (koefficient * TR) #Subtraktion från sista värde q(xn)

    return A_gles, HL


#------------------------Lösning uppgift a-----------

A = matris_A()
b = hl_uppg_a(TL, TR)

T = np.linalg.solve(A, b)
print(f"Uppgift T2a:\nTemperaturerna är {T}")


#------------------------Uppgift b-----------------

A = generell_matris_A(N, k, h)
b = hl_b(q, k, TL, TR, N, h, L)
lösn_b = np.linalg.solve(A,b)
print(f"Lösningen på uppgift T2b blir:\n{lösn_b}")

#------------------------Uppgift c-----------------
print("Uppgift C:")
A, HL = diskretisering_temperatur(N, q, k, TL, TR)
print (f"Matris A:\n{A.toarray()}")
print(f"Högerledet:\n{HL}")


#------------------------Uppgift d------------------

print("Uppgift T2d:")
A, HL = diskretisering_temperatur(100, q, k, TL, TR)
temperaturer = spsolve(A, HL)
approx_värdet = int(round(0.2 * 100))
värde = temperaturer[approx_värdet -1]
print(f"Det approximerade värdet är: {värde}")



x_punkter = np.linspace(0,1,99)
plt.xlabel("x")
plt.ylabel("Temperatur")
plt.plot(x_punkter, temperaturer, label ='Temperatur som funktion av x')
plt.legend()
plt.show()


#------------------------Uppgift e-----------------------


N_värden = [50, 100, 200, 400, 800, 1600]
T_exakt = 1.6379544
lista_fel = []


for N in N_värden:
    A, HL = diskretisering_temperatur(N, q, k, TL, TR)
    temperaturer = spsolve(A, HL)
    element_lösningsvekt = int(round(0.7 * N)-1) #Indexering för att alltid hamna vid x= 0.7
    T_approx = temperaturer[element_lösningsvekt]
    fel = np.abs(T_exakt-T_approx)
    lista_fel.append(fel)
    print(f"Beräknade felet är med N= {N}    Fel:{fel:.8f}")


for i in range(1, len(lista_fel)):
    kvot = lista_fel[i-1] / lista_fel[i]
    nogrannhetsordning = np.log2(kvot)
    print(f"Nogrannhetsordning: {nogrannhetsordning}")
