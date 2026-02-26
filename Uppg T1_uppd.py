import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def F(t, y, R, L, C):
    """Returnerar vektor på formen y' = F(t,y)"""
    q, i = y #y på vektorform
    dq_dt = i
    di_dt = (-R*i)/L - q/(C*L)

    return np.array([dq_dt, di_dt])

def hjälpfunktion_F(t,y,):
    """Hjälpfunktion för att nyttja värden på R,L,C"""
    R, L, C = 1, 2, 0.5
    return F(t, y, R, L, C)


def system_framåt_euler(F, tspan, U0, h):
    """Definierar euler framåt för ODE-system"""
    n = round(np.abs(tspan[1] - tspan[0])/h) #beräknar antal steg
    tk = np.zeros(n+1) #tidsvektor
    Uk = np.zeros((n + 1, len(U0)))

    tk[0] = tspan[0]
    Uk[0] = U0

    for k in range(n):
        Uk[k+1] = Uk[k] + h * F(tk[k], Uk[k])
        tk[k+1] = tk[k] + h
    return tk, Uk

def noggrannhetsordning(e, e_2):
    """Funktion som beräknar noggrannhetsordning"""
    return np.log(e / e_2) * 1 / np.log(2) #e_2 symboliserar felet med halva steglängden från föregående


#-------------------------UPPG C--------------

t_intervall = [0,20] #tidsintervallet för t
y0 = [1,0] #Startvärde Q0 = 1 och i = 0

R_L_C_1 = (1, 2, 0.5) # R=1, L=2, C=0.5 uppgift 1 på c
punkter = np.linspace(0, 20, 1000)
lösning_1 = solve_ivp(F, t_intervall, y0, method ='RK45', args = R_L_C_1, t_eval = punkter)

R_L_C_2 = (0, 2, 0.5) #R =0, L=2, C=0.5 uppgift 2 på c
lösning_2 = solve_ivp(F, t_intervall, y0, method ='RK45', args = R_L_C_2, t_eval = punkter)

plt.figure(1)
plt.plot(lösning_1.t, lösning_1.y[0],'b', label="Dämpad svängning")
plt.plot(lösning_2.t,lösning_2.y[0],'r', label = "Odämpad svängning")
plt.xlabel("t")
plt.ylabel("laddning q")
plt.title("RLC-Krets")
plt.legend()

plt.figure(2)
plt.plot(lösning_1.t, lösning_1.y[1],'b', label="Dämpad svängning")
plt.plot(lösning_2.t,lösning_2.y[1],'r', label = "Odämpad svängning")
plt.xlabel("t")
plt.ylabel("Ström i")
plt.title("RLC-Krets")
plt.legend()
plt.show()


#---------------------UPPG D-------------------
N_lista = [20,40,80,160]
t_intervall = [0,20] #tidsintervallet för t
y0 = [1,0]

#Laddningen q

plt.figure(3)
plt.plot(lösning_1.t, lösning_1.y[0],'b', label="Dämpad svängning")
for N in N_lista:
    h = (t_intervall[1] - t_intervall[0]) / N #beräknar steglängd för respektive N, t[1]=20 t[0]=0
    tk, Uk = system_framåt_euler(hjälpfunktion_F,t_intervall,y0,h)
    plt.plot(tk, Uk[:, 0], label = f"Euler för N={N}")

plt.title("Laddning q")
plt.xlabel("Tid t")
plt.ylabel("Laddning q")
plt.legend()
plt.ylim(-5,5)

#Strömmen i
plt.figure(4)
plt.plot(lösning_1.t, lösning_1.y[1],'r', label="Dämpad svängning")
for N in N_lista:
    h = (t_intervall[1] - t_intervall[0]) / N #beräknar steglängd för respektive N, t[1]=20 t[0]=0
    tk, Uk = system_framåt_euler(hjälpfunktion_F,t_intervall,y0,h)
    plt.plot(tk, Uk[:, 1], label = f"Euler för N={N}")

plt.title("Strömmen i")
plt.xlabel("Tid t")
plt.ylabel("Strömmen i")
plt.legend()
plt.ylim(-5,5)
plt.show()

#------------------------UPPG E------------------------

N_konvergens = [160, 320, 640, 1280] #N-värden för konvergensstudie, start med N=160 som är stabil
felen_q = []
felen_i = []

#Referenslösning given med solve_ivp
q_ref = lösning_1.y[0,-1] #Värdet q vid t=20
i_ref = lösning_1.y[1,-1] #Värdet i vid t=20

for N in N_konvergens:
    h = (t_intervall[1] - t_intervall[0]) / N
    tk , Uk = system_framåt_euler(hjälpfunktion_F, t_intervall, y0, h)
    fel_q = np.abs(Uk[-1,0] - q_ref)
    fel_i = np.abs(Uk[-1,1] - i_ref)
    felen_q.append(fel_q)
    felen_i.append(fel_i)
    print (f"Med N={N} ges felet:")
    print(f"e_q = {fel_q:.5f}\ne_i = {fel_i:.5f}")

for i in range(len(felen_q) -1):
    p_q = noggrannhetsordning(felen_q[i],felen_q[i+1])
    N_nu = N_konvergens[i]
    N_nästa = N_konvergens[i+1]
    print(f"Noggrannhetsordning mellan N={N_nu} och N={N_nästa} ger p_q={p_q:.5f}")

for i in range(len(felen_i) -1):
    p_i = noggrannhetsordning(felen_i[i],felen_i[i+1])
    N_nu = N_konvergens[i]
    N_nästa = N_konvergens[i+1]
    print(f"Noggrannhetsordning mellan N={N_nu} och N={N_nästa} ger p_i={p_i:.5f}")