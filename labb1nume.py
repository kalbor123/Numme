
import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 1.0
tau = 1e-10  # Tolerance

# The function f(x) for the beam
def f(x):
    return (8/3)*(x/L) - 3*(x/L)**2 + (1/3)*(x/L)**3 - (2/3)*np.sin(np.pi*x/L)

# The derivative f'(x) for Newton's method
def df(x):
    # Differentiating f(x) with respect to x
    return (8/(3*L)) - 6*x/(L**2) + (x**2)/(L**2) - (2*np.pi/(3*L))*np.cos(np.pi*x/L)

# The g(x) function for Fixed Point Iteration
def g(x):
    return (3*L/8) * (3*(x/L)**2 - (1/3)*(x/L)**3 + (2/3)*np.sin(np.pi*x/L))

# --- a) Plotting to find roots ---
x_plot = np.linspace(0, L, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x_plot, f(x_plot), label='f(x) - Deflection')
plt.axhline(0, color='black', linestyle='--')
plt.grid(True)
plt.title("Uppgift 1a: rötter av f(x)")
plt.legend()
plt.show() 
# Observation: Roots are at x=0, approx x=0.45, approx x=0.9, and x=1.

# --- c) Fixed Point Iteration ---
print("\n--- Fixed Point Iteration ---")
x_curr = 0.4  # Guess near the first internal root
error = 1.0
iter_fp = 0
errors_fp = []

while error > tau and iter_fp < 100:
    x_next = g(x_curr)
    error = abs(x_next - x_curr)
    errors_fp.append(error)
    print(f"Iter {iter_fp}: x = {x_next:.8f}, error = {error:.2e}")
    x_curr = x_next
    iter_fp += 1
root_fp = x_curr

# --- d) Newton's Method ---
print("\n--- Newtons metod ---")
# Choosing a root where Fixed Point likely fails (e.g., near 0.9)
x_curr = 0.3 
error = 1.0
iter_newt = 0
errors_newt = []

while error > tau and iter_newt < 100:
    x_next = x_curr - f(x_curr)/df(x_curr)
    error = abs(x_next - x_curr)
    errors_newt.append(error)
    print(f"Iter {iter_newt}: x = {x_next:.8f}, error = {error:.2e}")
    x_curr = x_next
    iter_newt += 1
root_newton = x_curr

# --- e) Convergence Comparison ---
# Run both on the SAME root (approx 0.48) to compare
x_start = 0.84
# Re-run FP
x_fp = x_start
err_fp_list = []
for _ in range(10):
    x_new = g(x_fp)
    err_fp_list.append(abs(x_new - x_fp))
    x_fp = x_new
    
# Re-run Newton
x_nw = x_start
err_nw_list = []
for _ in range(10):
    x_new = x_nw - f(x_nw)/df(x_nw)
    err_nw_list.append(abs(x_new - x_nw))
    x_nw = x_new

plt.figure()
plt.semilogy(err_fp_list, label='Fixed Point Error')
plt.semilogy(err_nw_list, label='Newton Error')
plt.xlabel('Iteration')
plt.ylabel('|x_{n+1} - x_n|')
plt.title('uppgift 1e: Convergence Speed Comparison')
plt.legend()
plt.grid(True)
plt.show()