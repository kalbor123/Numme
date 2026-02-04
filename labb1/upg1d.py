import numpy as np # hanterar många x-värden samtidigt


def f(x): #definition av f(x) vars nollställen söks
    return (8/3)*x - 3*x**2 + (1/3)*x**3 - (2/3)*np.sin(np.pi*x)

def df(x): # definierar derivatan f'(x)
    return ((8/3) - 6*x + x**2 - (2*np.pi/3)*np.cos(np.pi*x))

def g(x): #definierar fixpunktsfunktionen g(x) (används ej i upg d)
    return (3/8)*(3*x**2 - (1/3)*x**3 + (2/3)*np.sin(np.pi*x))

x = 0.3 # startgissning
tol = 1e-10 # tolerans för avbrottsvillkoret
diff = 1.0 # startvärde för skillnaden
it = 0 # räknare för antal iterationer
maxiter = 1000 # max antal tillåtna itterationer

while diff > tol and it < maxiter: # iterera tills konvergens eller max iterationer nås
    xnew = x - f(x)/df(x) # beräknar nästa värde med newtons metod
    diff = abs(xnew - x) # beräknar skillnaden mellan två iterationer
    print(it, xnew, diff) # skriver ut iteration, nytt x och skillnaden
    x = xnew # uppdaterar x till nya värdet
    it += 1 #ökar iterationsräknaren med 1

print("slutgiltigt x = ", x)
print("antal iterationer =", it)
print("sista diff =", diff)
print("|f(x)| =", abs(f(x)))
