import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

# --- Data från Tabell 1 [cite: 362, 363] ---
t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
y = np.array([421, 553, 709, 871, 1021, 1109, 1066, 929, 771, 612, 463, 374], dtype=float)

# --- Hjälpfunktioner ---
def eval_poly(c, x_vals, x_shift=0):
    """Beräknar polynomvärde manuellt: c0 + c1*x + ..."""
    res = np.zeros_like(x_vals)
    for i, coeff in enumerate(c):
        res += coeff * (x_vals - x_shift)**i
    return res

def newton_basis(t_points, t_eval):
    """Hjälpfunktion för Newtons ansats (basfunktioner)"""
    n = len(t_points)
    m = len(t_eval)
    N = np.ones((m, n))
    for j in range(1, n):
        N[:, j] = N[:, j-1] * (t_eval - t_points[j-1])
    return N

# ==========================================
# Uppgift 2a & 2b: Interpolation & Konditionstal
# ==========================================
print("--- Uppgift 2a & 2b ---")

# 1. Naiv ansats: 1, t, t^2 ...
A_naive = np.vander(t, increasing=True)
c_naive = la.solve(A_naive, y)
cond_naive = la.cond(A_naive, p=np.inf)

# 2. Centrerad ansats: t_m = medelvärde
tm = np.mean(t)
A_cent = np.vander(t - tm, increasing=True)
c_cent = la.solve(A_cent, y)
cond_cent = la.cond(A_cent, p=np.inf)

# 3. Newtons ansats (implementeras via matrisform för konditionstal)
# Matris N där N_ij = produkt(t_i - t_k) för k=0..j-1
n = len(t)
A_newt = np.zeros((n, n))
for j in range(n):
    for i in range(n):
        if j == 0:
            A_newt[i, j] = 1
        else:
            A_newt[i, j] = np.prod(t[i] - t[:j])
            
c_newt = la.solve(A_newt, y) # Dividerade differenser
cond_newt = la.cond(A_newt, p=np.inf)

print(f"Konditionstal (Maxnorm):")
print(f"  Naiv:      {cond_naive:.2e}")
print(f"  Centrerad: {cond_cent:.2e}")
print(f"  Newton:    {cond_newt:.2e}")

# Evaluera skillnad på fin grid (1000 punkter)
t_fine = np.linspace(1, 12, 1000)

# Naiv eval
p1 = eval_poly(c_naive, t_fine)
# Centrerad eval
p2 = eval_poly(c_cent, t_fine, x_shift=tm)
# Newton eval
N_fine = newton_basis(t, t_fine)
p3 = N_fine @ c_newt

diff_max = np.max([np.abs(p1 - p2), np.abs(p1 - p3), np.abs(p2 - p3)])
print(f"Maximal skillnad mellan ansatserna: {diff_max:.2e}")

# Plot 2a (bara en behövs enligt PM)
plt.figure(figsize=(10, 6))
plt.plot(t, y, 'ro', label='Data')
plt.plot(t_fine, p2, 'b-', label='Interpolation (Centrerad)')
plt.title("Uppgift 2a: Interpolation (Grad 11)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# Uppgift 2c, d, e, f: Minstakvadrat (April-Aug)
# ==========================================
print("\n--- Uppgift 2c, d, e, f ---")

# Data urval: April (4) till Augusti (8). Index 3 till 7 i Python.
idx = slice(3, 8) 
t_sub = t[idx]
y_sub = y[idx]
t_plot = np.linspace(3, 9, 100) # För att plotta kurvorna snyggt

def solve_lsq(A, y_vec, model_name):
    # Normalekvationerna: A.T * A * c = A.T * y
    ATA = A.T @ A
    ATy = A.T @ y_vec
    c = la.solve(ATA, ATy)
    
    # Konditionstal för normalekvationsmatrisen (Uppgift 2f)
    cond_num = la.cond(ATA, p=np.inf)
    
    # Beräkna fel (kvadrat)
    resid = la.norm(y_vec - A @ c)**2
    
    print(f"Modell: {model_name}")
    print(f"  Konditionstal (ATA): {cond_num:.2e}")
    print(f"  Minstakvadratfel: {resid:.2f}")
    return c

plt.figure(figsize=(10, 6))
plt.plot(t_sub, y_sub, 'ko', label='Data (Apr-Aug)', zorder=5)

# 2c: Andragradspolynom (1, t, t^2)
A_quad = np.vander(t_sub, 3, increasing=True)
c_quad = solve_lsq(A_quad, y_sub, "2c: Andragrad")
y_quad = eval_poly(c_quad, t_plot)
plt.plot(t_plot, y_quad, '--', label='2c: Andragrad')

# 2d: Tredjegradspolynom (1, t, t^2, t^3)
A_cubic = np.vander(t_sub, 4, increasing=True)
c_cubic = solve_lsq(A_cubic, y_sub, "2d: Tredjegrad")
y_cubic = eval_poly(c_cubic, t_plot)
plt.plot(t_plot, y_cubic, '-.', label='2d: Tredjegrad')

# 2e: Trigonometrisk
# p(t) = d0 + d1*cos(wt) + d2*sin(wt)
omega = 2 * np.pi / 12
A_trig = np.column_stack([np.ones_like(t_sub), np.cos(omega*t_sub), np.sin(omega*t_sub)])
c_trig = solve_lsq(A_trig, y_sub, "2e: Trigonometrisk")
y_trig = c_trig[0] + c_trig[1]*np.cos(omega*t_plot) + c_trig[2]*np.sin(omega*t_plot)
plt.plot(t_plot, y_trig, '-', label='2e: Trigonometrisk')

plt.title("Uppgift 2c-e: Minstakvadratanpassning")
plt.legend()
plt.grid(True)
plt.show()