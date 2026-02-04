import numpy as np

t = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
f = np.array([12.00, 15.10, 19.01, 23.92, 30.11, 37.90, 47.70, 60.03, 75.56])

def trapets(f, n, ab):
    a = ab[0] # undre integralgräns
    b = ab[1] # övre integralgräns
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b)) #yttre punkter
    for i in range(1, n):  # lägger till inre punkter
        x = a + h * i
        s += f(x)

    return h * s
def uppgift_3a():
    def f(x):
        return x**3 * np.exp(x)

    T = trapets(f, 1000, [0, 2])
    print("T =", T)
    print()

def uppgift_3b():
    def f(x):
        return x**3 * np.exp(x)

    I = 6 + 2*np.exp(2)

    n_värden = [50, 100, 200, 400, 800]

    for n in n_värden:
        T = trapets(f, n, [0,2])
        fel = abs(I - T)
        print("n =", n, "och fel =", fel)

def trapets_tabell(f_värden, h):
    return h * (0.5*f_värden[0] + sum(f_värden[1:-1]) + 0.5*f_värden[-1])
    # '0' betyder första '-1' betyder sista
    # sum(f[1:-1]) summerar alla inre funktionsvärden


def uppgift_3c():

    h = 1

    T = trapets_tabell(f, h)
    print()
    print("i uppgift c är T =", T)
    print()

def uppgift_3d():
    h_värden = [1, 2, 4, 8]

    T_före = None

    for h in h_värden:
        f_h = f[::h]
        Th = trapets_tabell(f_h, h)
        print("h =", h, "Th =", Th)

        if T_före is not None:
            print("skilllnad = ", abs(T_före - Th))
            print()

        T_före = Th

def uppgift_3e():

    T1 = trapets_tabell(f, 1)
    T2 = trapets_tabell(f[::2], 2)

    R = (4*T1 - T2) / 3 #Rickardsson
    S = (1/3) * (f[0] + f[-1] + 4*sum(f[1:-1:2]) + 2*sum(f[2:-1:2])) #simpson

    print("T1 =", T1)
    print("T2 =", T2)
    print("Richardson R =", R)
    print("Simpson S =", S)
    print("|R - S| =", abs(R - S))
    print()


def uppgift_3f():
    x = t - 2014          # x = (t-2014)
    y = np.log(f)         # y = ln(f)

    A = np.column_stack((np.ones_like(x), x))   # kolumner: [1, x]
    c, b = np.linalg.lstsq(A, y, rcond=None)[0] # löser y ≈ c + b x

    a = np.exp(c)         # c = ln(a) => a = e^c

    print("a =", a)
    print("b =", b)
    print()

def uppgift_3g():
    # --- ta fram a och b precis som i 3f ---
    x = t - 2014
    y = np.log(f)
    A = np.column_stack((np.ones_like(x), x))
    c, b = np.linalg.lstsq(A, y, rcond=None)[0]
    a = np.exp(c)

    # --- uppskatta f(2023) ---
    f2023 = a * np.exp(b*(2023 - 2014))

    # --- lägg till i tabellen ---
    t_ext = np.append(t, 2023.0)
    f_ext = np.append(f, f2023)

    # --- skatta integralen 2014..2023 med trapets på tabell ---
    E = trapets_tabell(f_ext, 1)

    print("f(2023) =", f2023)
    print("Energi 2014-2023 =", E)

    print("Villkor 1 (f2023 > 100):", f2023 > 100)
    print("Villkor 2 (E > 350):", E > 350)









if __name__ == "__main__":
    uppgift_3a()
    uppgift_3b()
    uppgift_3c()
    uppgift_3d()
    uppgift_3e()
    uppgift_3f()
    uppgift_3g()



