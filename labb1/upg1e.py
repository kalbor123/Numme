import numpy as np #hanterar många x-värden samtidigt
import matplotlib.pyplot as plt #ritar grafer

def f(x):
    return (8/3)*x - 3*x**2 + (1/3)*x**3 - (2/3)*np.sin(np.pi*x)

def df(x):
    return ((8/3) - 6*x + x**2 - (2*np.pi/3)*np.cos(np.pi*x))

def g(x):
    return (3/8)*(3*x**2 - (1/3)*x**3 + (2/3)*np.sin(np.pi*x))

tol = 1e-10
maxiter = 10
x0 = 0.84
# ----- Fixpunktsmetoden -----
x = x0
diffs_fp = []
it = 0
diff = 1.0

while diff > tol and it < maxiter:
    xnew = g(x)
    diff = abs(xnew - x)
    diffs_fp.append(diff)
    x = xnew
    it += 1

# ----- Newtons metod -----
x = x0
diffs_newton = []
it = 0
diff = 1.0
while diff > tol and it < maxiter:
    xnew = x - f(x)/df(x)
    diff = abs(xnew - x)
    diffs_newton.append(diff)
    x = xnew
    it += 1


#--- Plottar ---
plt.semilogy(diffs_fp, label="fixpunkt")
plt.semilogy(diffs_newton, label="newton")
plt.xlabel("iteration n")
plt.ylabel(r"$|x_{n+1}-x_n|$")
plt.legend()
plt.grid(True)
plt.show()

