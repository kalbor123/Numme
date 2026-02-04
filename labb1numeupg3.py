import numpy as np
import matplotlib.pyplot as plt

# --- Data från Tabell 2 [cite: 419, 420] ---
years = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022], dtype=float)
power = np.array([12.00, 15.10, 19.01, 23.92, 30.11, 37.90, 47.70, 60.03, 75.56])

# --- 3a: Funktion för Trapetsregeln ---
def trapets(func_values, h):
    """
    Beräknar integralen med trapetsregeln givet funktionsvärden och steglängd h.
    T_h = h * (0.5*f0 + f1 + ... + fn-1 + 0.5*fn)
    """
    s = np.sum(func_values) - 0.5*func_values[0] - 0.5*func_values[-1]
    return h * s

# Verifiering på testintegral x^3 * e^x från 0 till 2 [cite: 426]
print("--- Uppgift 3a: Verifiering ---")
x_test_fine = np.linspace(0, 2, 1000)
y_test_fine = x_test_fine**3 * np.exp(x_test_fine)
exact_I = 6 + 2 * np.exp(2)
# Testa med h=0.002 (approx)
calc_I = trapets(y_test_fine, 2/999)
print(f"Exakt: {exact_I:.4f}, Beräknad: {calc_I:.4f}")

# --- 3c: Skatta integralen för solceller (h=1) ---
print("\n--- Uppgift 3c ---")
h1 = 1.0
T_h1 = trapets(power, h1)
print(f"Total energi 2014-2022 (h=1): {T_h1:.4f} kWår")

# --- 3d: Konvergensstudie på data ---
# Vi har data för h=1. Vi simulerar h=2 genom att ta vartannat värde.
# Vi simulerar h=4 genom att ta vart fjärde värde.
print("\n--- Uppgift 3d: Konvergensstudie ---")
power_h2 = power[::2] # Vartannat element
h2 = 2.0
T_h2 = trapets(power_h2, h2)

power_h4 = power[::4] # Vart fjärde element
h4 = 4.0
T_h4 = trapets(power_h4, h4)

print(f"T(h=1): {T_h1:.4f}")
print(f"T(h=2): {T_h2:.4f}")
print(f"T(h=4): {T_h4:.4f}")

# Uppskatta fel och kvot
diff_1 = abs(T_h1 - T_h2) # ~ e_2h
diff_2 = abs(T_h2 - T_h4) # ~ e_4h
kvot = diff_2 / diff_1
print(f"Kvot (e_4h / e_2h): {kvot:.2f} (Teoretiskt ≈ 4 för Trapets)")

# --- 3e: Richardson & Simpson ---
print("\n--- Uppgift 3e ---")
# Richardson: R = T_h + (T_h - T_2h) / 3 (för p=2)
R_extrap = T_h1 + (T_h1 - T_h2) / 3
print(f"Richardsonextrapolation: {R_extrap:.4f}")

# Simpsons regel: S = h/3 * (f0 + 4*sum(udda) + 2*sum(jämna) + fn)
odd_sum = np.sum(power[1:-1:2])  # Index 1, 3, 5...
even_sum = np.sum(power[2:-2:2]) # Index 2, 4, 6...
S_simpson = (h1 / 3) * (power[0] + power[-1] + 4*odd_sum + 2*even_sum)
print(f"Simpsons regel:          {S_simpson:.4f}")

# --- 3f: Exponentiell modell (Linjärisering) ---
print("\n--- Uppgift 3f & 3g ---")
# Modell: f(t) = a * exp(b*(t - 2014))
# Linjärisering: ln(f) = ln(a) + b*(t - 2014)
# y_lin = c0 + c1 * x_lin
t_shifted = years - 2014
y_log = np.log(power)

# Minstakvadratanpassning (Grad 1 på log-data)
A_lin = np.vstack([np.ones(len(t_shifted)), t_shifted]).T
c_lin = np.linalg.lstsq(A_lin, y_log, rcond=None)[0]

ln_a = c_lin[0]
b = c_lin[1]
a = np.exp(ln_a)

print(f"Modellparametrar: a = {a:.4f}, b = {b:.4f}")

# --- 3g: Prognos 2023 ---
# Prediktera effekt år 2023 (t - 2014 = 9)
f_2023 = a * np.exp(b * 9)
print(f"Prognos effekt 2023: {f_2023:.2f} kW")

# Lägg till i tabell och integrera (h=1)
years_ext = np.append(years, 2023)
power_ext = np.append(power, f_2023)
total_energy_2023 = trapets(power_ext, h1)
print(f"Total energi 2014-2023: {total_energy_2023:.2f} kWår")

# Villkor för succé [cite: 442]
cond1 = f_2023 > 100
cond2 = total_energy_2023 > 350

if cond1 or cond2:
    print("SLUTSATS: Pilotprojektet anses lyckat!")
else:
    print("SLUTSATS: Pilotprojektet nådde inte målen.")

# Plot för att visualisera
plt.figure()
plt.plot(years, power, 'o', label='Mätdata')
t_plot = np.linspace(2014, 2023, 100)
f_plot = a * np.exp(b * (t_plot - 2014))
plt.plot(t_plot, f_plot, label='Exp. modell')
plt.plot(2023, f_2023, 'r*', markersize=10, label='Prognos 2023')
plt.title("Solcellseffekt & Prognos")
plt.legend()
plt.grid()
plt.show()