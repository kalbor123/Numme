import numpy as np
import matplotlib.pyplot as plt

def f(t,y):
    """Definierar f(t,y)"""
    return 1 + t - y

def exakt_lösning(t):
    """Den exakta lösningen till differentialekvationen"""
    y = np.exp(-t) + t
    return y


def riktningsfält(f,tmin,tmax,ymin,ymax,density,scale):
    """Skissar riktningsfält uppg a"""
    xs = np.linspace(tmin, tmax, density)
    ys = np.linspace(ymin,ymax, density)
    X, Y = np.meshgrid(xs,ys)

    S = f(X,Y)
    U = np.ones_like(S)
    V = S
    L = np.hypot(U,V)
    U /= L
    V /= L

    plt.figure(figsize =(5,5))
    plt.quiver(X,Y,U,V, scale=scale)

    t_vec = np.linspace(tmin, tmax, 200)

    y_vec = exakt_lösning(t_vec)
    plt.plot(t_vec, y_vec, color = 'blue', label ='f(t)')

    plt.xlim(tmin, tmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Riktningsfält: dy/dt = 1 + t - y")


def framåt_euler(f, tspan, y0, h):
    """Definierar Euler framåt"""
    a , b = tspan[0],tspan[1] #Definierar tidsintervall t
    n = round(np.abs(b - a)/h) #Beräknar antalet steg
    t = np.linspace(a, b, n+1)

    y = np.zeros(n+1)

    #Begynnelsvillkor
    y[0] = y0
    for k in range(n):
        y[k+1] = y[k] + h*f(t[k], y[k])
    return t, y


def beräkna_fel(appr_y, exakt_y):
    fel = np.abs(appr_y - exakt_y)
    return fel


#-------------------------------------UPPG F2-----------------------


def halvera_h():
    h_lista = [0.2,0.1,0.05,0.025,0.0125]
    tspan = [0, 1.2]
    y0 = 1
    lista_lösningar = []

    for h in h_lista:
        #Beräknar respektive värden på t och y 
        t_res , y_res = framåt_euler(f, tspan, y0, h)
        #Beräknar värden vid sluttiden T
        t_slut = t_res[-1]
        y_slut = y_res[-1]
        lista_lösningar.append(y_slut)
        print(f"Steglängd: {h}  Sluttid T={t_slut:<6.2f} Slutvärde y= {y_slut:<6.6f}")

    return h_lista, lista_lösningar

def noggrannhetsordning(e, e_2):
    """Funktion som beräknar noggrannhetsordning"""
    return np.log(e / e_2) * 1 / np.log(2) #e_2 symboliserar fel vid halva steglängden


def main():
    #Mellan t = 0 och t = 1.2, y: 0-2
    riktningsfält(f, 0, 1.2, 0, 2, density= 25, scale = 25)
    tspan = np.array([0,1.2])
    y0 = 1
    h = 0.1
    t_res, y_res = framåt_euler(f, tspan, y0, h) # Lagrar resultat för t och y med stegen h
    plt.plot(t_res, y_res,'r', label = 'Euler framåt')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Approximativ lösning dy/dt = 1 + t - y')
    plt.legend()
    plt.grid(True)
    plt.show()

    #----------------------Uppgift C---------------

    y_appr = y_res[-1] #Approximativa värdet med T=1.2
    y_exakt = exakt_lösning(1.2) #Beräknar exakta värdet med T=1.2
    fel = beräkna_fel(y_appr, y_exakt)
    print(f"Felet i approximationen uppg F1c är: {fel:.4f}")


    #--------------------UPPGIFT F2--------------------
    print("Uppgift F2:")
    print("Uppgift F2a:")
    h_värden, y_värden = halvera_h()
    lista_fel =[]

    print("Uppgift F2b och c:")
    for h, y in zip(h_värden, y_värden): #Matchar steglängd h med y-värde baserat på index
        fel = beräkna_fel(y, y_exakt)
        lista_fel.append(fel)
        print(f"Steglängd: {h:<6} y-värde: {y:<4f}, Fel= {fel:.2e}")

    for i in range(len(lista_fel) -1):
        p = noggrannhetsordning(lista_fel[i],lista_fel[i+1])
        print(f"Noggrannhetsordning: {p:.3f}")



if __name__ == "__main__":
    main()

