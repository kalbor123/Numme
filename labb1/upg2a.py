import numpy as np
import matplotlib.pyplot as plt

# --- Data (x=t, y=soltid) ---
t = np.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype=float) # skapar vektor t med månaderna 1-12 som flyttal
y = np.array([421,553,709,871,1021,1109,1066,929,771,612,463,374], dtype=float) #skapar vektorn y,
# soltid för respektive månad.

# --- (c) Minstakvadrat grad 2, april–augusti ---
t_c = t[3:8]   # månader 4..8 (april–augusti)
y_c = y[3:8]
    

tm = np.mean(t) #medelvärde av t
s = t - tm # centrerade t-värden

n = len(t)    #antal datapunkter (n = 12) (len() returnerar antalet element)
grad = n - 1     #grad för interpolationspolynom, dvs 12-1 = 11

# ------naiv ansats------
# dvs söker polynom p(t) = c0 + c1*t + c2*t^2 + ... + c_(n-1)*t^(n-1).

A = np.zeros((n, n))  #skapar en tom nxn-matris (dvs 12x12) i detta fall
A[:, 0] = 1.0 #dvs varje rad börjar med 1


for j in range(1, n):
    A[:, j] = A[:, j-1] * t   #bygger matrisen innan vi vet polynomet. Här börjar j på kolumn 2, (dvs på kolumn1)
    #eftersom kolumn 0 är 1.
    # ":" betyder alla rader, j är kolumn nummer j. A[:, j] betyder hela kolumn j i matris A
    # Exempelvis om j = 2, ta kolumn 2 i matris A.
    # A[:, j-1] * t gör att varje t i kolumn j blir t^j eftersom t^j = t^(j-1) * t
    # ALLTSÅ: A[:, j] = A[:, j-1] * t betyder, fyll i hela kolumn j med potens
# --- Lös linjära systemet för koefficinterna c ---
b = y.reshape(n, 1) # sätter y = b i högerled som kolumnvekotor
c = np.linalg.solve(A, b) # gkoefficienter c0..c_(n-1), och löser det linjära ekvationssystemet Ac = b

# ---- Centrerad ansats ----
# Här byggs interpolationspolynom med den centrerade variabeln istället.
# dvs s = t - tm

A2 = np.zeros((n, n))  #skapar en ny nxn matris för den centrerade ansatsen
A2[:, 0] = 1.0  # första kolumnen är (t-tm^0) = 1 för alla rader

for j in range(1, n): #kolumn j blir nästa potens av (t-tm)
    A2[:, j] = A2[:, j-1] * s # samma som för A fast i  centrerad.

c2 = np.linalg.solve(A2, b) # löser det linjära ekvationssysteme A2*c2 = b


# ---- Newtons ansats ----

A3 = np.zeros((n, n))
A3[:, 0] = 1.0

for j in range(1, n):
    A3[:, j] = A3[:, j-1] * (t -t[j-1])

c3 = np.linalg.solve(A3, b)


# ---- konditionstal ---
kA  = np.linalg.cond(A,  np.inf)
kA2 = np.linalg.cond(A2, np.inf)
kA3 = np.linalg.cond(A3, np.inf)

print("cond_inf(A)  =", kA)
print("cond_inf(A2) =", kA2)
print("cond_inf(A3) =", kA3)



# --- Utvärdera polynomet för plot ---
tt = np.linspace(1, 12, 1000)  # skapar 1000 jämnt fördelade t-värden mellan 1 och 12
pp = np.zeros_like(tt)  # är en vektor som har samma format som tt men är fylld med nollor
for j in range(n):
    pp += c[j] * tt**j # fyller pp med polynomets värde i varje punkt.

# ---- centrerad utvärdering-----
ss = tt - tm          #centrerade t-värden
pp2 = np.zeros_like(tt)  # tom vekotr för centrerade polynomvärden

for j in range(n):
    pp2 += c2[j] * ss**j

# ---- utvärdera newtons polynom p3 ----

pp3 = np.zeros_like(tt)
basis = np.ones_like(tt)   # håller aktuell Newton-basterm börjar med 1

pp3 += c3[0] * basis # c0 * 1

for j in range(1, n):
    basis = basis * (tt - t[j-1]) #uppdaterar basen till bas*(tt - t_j)
    pp3 += c3[j] * basis #lägger till c_j * bas

# --- jämför polynomen numeriskt ---
print("max |p1 - p2| =", np.max(np.abs(pp - pp2)))
print("max |p1 - p3| =", np.max(np.abs(pp - pp3)))
print("max |p2 - p3| =", np.max(np.abs(pp2 - pp3)))


# ----- plotta data + polynom ------

plt.figure() #skapar ny tom figur
plt.plot(t, y, "o", label="data") #plottar mätpunkterna, t= xvärden y= soltid "o" gör att de plottas som punkter och ej linje
plt.plot(tt, pp, "-", label="interpolationspolunom (naiv)") # plottar polynomkurvan som en linje
plt.xlabel("Månad t")
plt.ylabel("Soltid (min)")
plt.grid(True) # visar rutnät
plt.legend()
plt.show()

print("graden blir", grad)