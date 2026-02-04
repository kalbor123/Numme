import numpy as np
import matplotlib.pyplot as plt


t = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
y = np.array([421, 553, 709, 871, 1021, 1109, 1066, 929, 771, 612, 463, 374])

#Skapar arrayer för minsta kvadratmetod
t_k = t[3 : 8]
y_k = y[3 : 8]


def naiv_matris(t):
    """Bildar matris för naiv ansats"""
    return np.vander(t, increasing = True)

def centrerad_matris(t):
    """Centrerad ansats"""
    medelvärde = np.average(t)
    z = t-medelvärde
    return np.vander(z, increasing = True)

def newtons_matris(t):
    """Skapar matris för Newtons ansats"""
    n = len(t)
    matris = np.zeros((n,n))
    matris[:, 0] = 1 #1:or i första kolumnen

    for j in np.arange(1,n):
        for i in np.arange(j,n):
            matris[i, j] = matris[i, j-1] * (t[i] - t[j-1])
    return matris

def beräkna_maxnorm(t):
    return np.max(abs(t))

def beräkna_konditionstal(matris):
    """Beräknar konditionstalet genom att mulitplicera matris med matrisinversen"""
    konditionstal = np.linalg.cond(matris)
    return f"{konditionstal:.2e}"

def minsta_kvadrat_matris_2(t_k):
    """Bildar matris A för minsta kvadrat anpassning grad 2 """
    n = len(t_k)
    kolumn_1 = np.ones(n)
    kolumn_2 = t_k
    kolumn_3 = t_k**2

    A = np.column_stack([kolumn_1,kolumn_2,kolumn_3])
    return A


def minsta_kvadrat_metod_2(t_k, y_k):
    """Löser ekvationssystem"""
    A = minsta_kvadrat_matris_2(t_k)
    A_T = np.transpose(A)
    vänsterled = A_T @ A
    högerled = A_T @ y_k
    return np.linalg.solve(vänsterled, högerled)

def minsta_kvadrat_matris_3(t_k):
    """Bildar matris A för minsta kvadratanpassning grad 3"""
    n = len(t_k)
    kolumn_1 = np.ones(n)
    kolumn_2 = t_k
    kolumn_3 = t_k**2
    kolumn_4 = t_k**3

    A= np.column_stack([kolumn_1,kolumn_2,kolumn_3,kolumn_4])
    return A

def minsta_kvadrat_metod_3(t_k, y_k):
    """Löser ekvationssystem"""
    A = minsta_kvadrat_matris_3(t_k)
    A_T = np.transpose(A)
    vänsterled = A_T @ A
    högerled = A_T @ y_k
    return np.linalg.solve(vänsterled, högerled)


t_punkter = np.linspace(1,12, 1000) #1000 punkter med lika avstånd mellan t=1 och t=12

A = naiv_matris(t)
c_naiv = np.linalg.solve(A, y) #Löser koefficienter till ekv.systemet
#Evaluering med naiv ansats
naiv_evaluering = np.zeros(1000) #array med 1000 element, enbart 0or
for i in range(len(c_naiv)):
    naiv_evaluering += c_naiv[i] * t_punkter**i


V = centrerad_matris(t)
t_medel = np.average(t) #Beräknar medelvärdet för centrerad ansats
c_centrerad = np.linalg.solve(V, y) #Löser koefficienter för ekv.systemet
#Evaluering med centrerad ansats
centrerad_evaluering = np.zeros(1000)
for i in range(len(c_centrerad)):
    centrerad_evaluering += c_centrerad[i] * (t_punkter - t_medel)**i


#Evaluering med Newtons metod
N = newtons_matris(t)
c_newtons = np.linalg.solve(N, y) #Löser ekv.systemet
newtons_evaluering = np.zeros(1000)
for i in range (len(c_newtons)):
    term_newtons = c_newtons[i]
    #Inför j för hantering av vilken stödpunkt
    for j in range(i):
        term_newtons *= (t_punkter - t[j])
    newtons_evaluering += term_newtons


#Beräkning av skillander:
#Differens naiv och centrerad evaluering
differens_naiv_centrerad = beräkna_maxnorm(centrerad_evaluering - naiv_evaluering)
print(f"Differens naiv och centrerad:\n{differens_naiv_centrerad}")

#Differens Newtons och Centrerad evaluering
differens_newtons_centrerad = beräkna_maxnorm(newtons_evaluering - centrerad_evaluering)
print(f"Differens Newtons och centrerad:\n{differens_newtons_centrerad}")

#Differens Newtons och naiv evaluering
differens_newtons_naiv = beräkna_maxnorm(newtons_evaluering - naiv_evaluering)
print(f"Differens Newtons och naiv:\n{differens_newtons_naiv}")

#Plottar polynomet från Newtons metod
plt.figure()
plt.plot(t, y, "o", label="Data")
plt.plot(t_punkter, naiv_evaluering, '-', label="Interpolationspolynom (naiv)")
plt.xlabel("Månader")
plt.ylabel("Soltid (Minuter)")
plt.legend()
plt.grid(True)
plt.show()

#Beräknar konditionstal för de olika ansatsmatriserna
k_naiv_matris = beräkna_konditionstal(A)
k_centrerad_matris = beräkna_konditionstal(V)
k_newtons_matris = beräkna_konditionstal(N)

print(f"Konditionstalen är:\nK-naiv: {k_naiv_matris}, K-centrerad: {k_centrerad_matris}, K-Newtons {k_newtons_matris}")


#-----------UPPGIFT 2C------------

A_grad_2 = minsta_kvadrat_matris_2(t_k)   # sparar matris använding i 2f
c = minsta_kvadrat_metod_2(t_k, y_k) #Värdet på koefficienter lösta från ekv.system
modell_grad_2 = np.linspace(4, 8, 100)
y_värden = c[0] + c[1]*modell_grad_2 + c[2]*modell_grad_2**2
#Skapar plot
plt.figure(2)
plt.plot(t_k, y_k, "o", label="data (apr–aug)")
plt.xlabel("Månader")
plt.ylabel("Soltid (Minuter)")
plt.title("Minsta kvadratanpassning (apr–aug)")
plt.plot(modell_grad_2, y_värden, 'b', label="grad 2")



#----------UPPGIFT 2D--------------

x =  minsta_kvadrat_metod_3(t_k,y_k)
A_grad_3 = minsta_kvadrat_matris_3(t_k)   # sparar matrir för normalekvationerna för 2f
c = minsta_kvadrat_metod_3(t_k,y_k)
modell_grad_3 = np.linspace(4, 8, 100)
y_värden = c[0] + c[1]*modell_grad_3 + c[2]*modell_grad_3**2 + c[3]*modell_grad_3**3
plt.figure(2)
plt.plot(modell_grad_3, y_värden, 'g', label="grad 3")
plt.grid(True)
plt.legend()



#------------UPPGIFT 2E--------------
# p(t) = d0 + d1*cos(wt) + d2*sin(wt)
# w = omega

omega = 2*np.pi/12

A_trig = np.column_stack([np.ones(len(t)), np.cos(omega*t), np.sin(omega*t)])

A_T = np.transpose(A_trig)
vänsterled = A_T @ A_trig
högerled = A_T @ y
d = np.linalg.solve(vänsterled, högerled)

modell_trig = np.linspace(1, 12, 1000)
y_trig = d[0] + d[1]*np.cos(omega*modell_trig) + d[2]*np.sin(omega*modell_trig)

plt.figure(2)
plt.plot(t, y, "o", label="data")
plt.plot(modell_trig, y_trig, "-", label="Minsta kvadrat (trig)")
plt.xlabel("Månad")
plt.ylabel("Soltid (Minuter)")
plt.title("Minsta kvadratanpassning: trigonometrisk modell")
plt.grid(True)
plt.legend()

#------------UPPGIFT 2F--------------

k_grad_2 = np.linalg.cond(A_grad_2.T @ A_grad_2, np.inf)
k_grad_3 = np.linalg.cond(A_grad_3.T @ A_grad_3, np.inf)
k_trig = np.linalg.cond(A_trig.T @ A_trig, np.inf)

print("Konditionstal för normalekvationernas matris (A^T A):")
print(f"Grad 2: {k_grad_2:.2e}")
print(f"Grad 3: {k_grad_3:.2e}")
print(f"Trig:   {k_trig:.2e}")

plt.show()

