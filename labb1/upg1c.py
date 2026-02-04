import numpy as np # hanterar många x-värden samtidigt

def f(x): #definition av f(x) vars nollställen söks
    return (8/3)*x - 3*x**2 + (1/3)*x**3 - (2/3)*np.sin(np.pi*x)

def g(x): #definierar fixpunktsfunktionen g(x)
    return (3/8)*(3*x**2 - (1/3)*x**3 + (2/3)*np.sin(np.pi*x))

x = 0.5 #startgissning för iterationen
tol = 1e-10 # Tolerans för avbrottsvillkoret
diff = 1.0 #startvärde för skillnaden x_n+1 och x_n
it = 0  #räknare av iterationer. Startar på 0.
maxiter = 1000

while diff > tol and it < maxiter: # itererar tills konvergens eller max iterationer nås
    xnew = g(x) #beräknar nästa fixpunkt
    diff = abs(xnew - x) # beräknar skillnaden mellan två iterationer
    print(it, xnew, diff) # skriver ut iterationen, nytt x och skillnaden
    x = xnew # uppdaterar x till nya värdet
    it += 1 # ökar iterationsräknaren

print("slutgiltigt x = ", x)
print("antal iterationer =", it)
print("sista diff =", diff)
print("|f(x)| =", abs(f(x)))
