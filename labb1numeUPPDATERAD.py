import numpy as np
import matplotlib.pyplot as plt

# Konstanter
tau = 1e-10   # Toleransen

# Funktionen f(x) för balkens utböjning (med L=1 insatt)
def f(x):
    # Ekvation (1) med L=1
    return (8/3)*(x) - 3*(x)**2 + (1/3)*(x)**3 - (2/3)*np.sin(np.pi*x)

# Derivatan f'(x) för Newtons metod (med L=1 insatt)
def df(x):
    # Deriverar f(x) med avseende på x
    return (8/3) - 6*x + (x**2) - (2*np.pi/3)*np.cos(np.pi*x)

# Definierar fixpunktsfunktionen g(x) (med L=1 insatt)
def g(x):
    # Ekvation (2) med L=1
    return (3/8)*(3*x**2 - (1/3)*x**3 + (2/3)*np.sin(np.pi*x))

# --- a) Plotta för att hitta rötter ---
# Här ändrat från L till 1
x_plot = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, f(x_plot), label='f(x) - Utböjning')
plt.axhline(0, color='black', linestyle='--')
plt.grid(True)
plt.title("Uppgift 1a: Rötter av f(x) (L=1)")
plt.legend()
plt.show() 
# Observation: Rötter finns vid x=0, ca x=0.45, ca x=0.82 och x=1.

# --- c) Fixpunktsiteration ---
print("\n--- Fixpunktsiteration ---")
x_nu = 0.45      # Startgissning nära den första inre roten (där fixpunkt fungerar)
fel = 1.0        # Startvärde för felet
iter_fp = 0
fel_lista_fp = []
max_iter = 300   # Maxantal iterationer

while fel > tau and iter_fp < max_iter:
    x_nasta = g(x_nu)
    fel = abs(x_nasta - x_nu)
    fel_lista_fp.append(fel)
    print(f"Iter {iter_fp}: x = {x_nasta:.8f}, fel = {fel:.2e}")
    x_nu = x_nasta
    iter_fp += 1
rot_fp = x_nu

# --- d) Newtons Metod ---
print("\n--- Newtons metod ---")
# Väljer en rot där fixpunktsmetoden troligen misslyckas (t.ex. nära 0.3)
x_nu = 0.3 
fel = 1.0
iter_newt = 0
fel_lista_newt = []

while fel > tau and iter_newt < 100:
    x_nasta = x_nu - f(x_nu)/df(x_nu)
    fel = abs(x_nasta - x_nu)
    fel_lista_newt.append(fel)
    print(f"Iter {iter_newt}: x = {x_nasta:.8f}, fel = {fel:.2e}")
    x_nu = x_nasta
    iter_newt += 1
rot_newton = x_nu

# --- e) Jämförelse av konvergens ---
# Kör båda på SAMMA rot (ca 0.82) för att kunna jämföra
x_start = 0.84

# Kör om Fixpunkt för plotten
x_fp = x_start
err_fp_lista = []
for _ in range(10):
    x_ny = g(x_fp)
    err_fp_lista.append(abs(x_ny - x_fp))
    x_fp = x_ny
    
# Kör om Newton för plotten
x_nw = x_start
err_nw_lista = []
for _ in range(10):
    x_ny = x_nw - f(x_nw)/df(x_nw)
    err_nw_lista.append(abs(x_ny - x_nw))
    x_nw = x_ny

plt.figure()
plt.semilogy(err_fp_lista, label='Fixpunktsfel')
plt.semilogy(err_nw_lista, label='Newtonfel')
plt.xlabel('Iteration')
plt.ylabel('|x_{n+1} - x_n|')
plt.title('Uppgift 1e: Jämförelse av konvergenshastighet')
plt.legend()
plt.grid(True)
plt.show()