#Para resolver numéricamente el sistema del motor DC,
#se utilizó el método de Euler mediante el siguiente código
#El código implementa el método de Euler para resolver el sistema de ecuaciones
#diferenciales del motor DC. Se simula la evolución de la corriente
#y la velocidad angular en el tiempo bajo un voltaje constante
#El código implementado en Python corresponde directamente al modelo matemático del 
#motor DC obtenido en los días anteriores. A partir de las ecuaciones diferenciales del 
#sistema, se aplicó el método de Euler para aproximar la solución numérica, permitiendo
#analizar la evolución de la corriente y la velocidad angular en el tiempo.
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del motor DC
R = 2.0     # Resistencia (ohmios)
L = 0.5     # Inductancia (henrios)
Ke = 0.01   # Constante FEM (V*s/rad)
Kt = 0.01   # Constante de torque (N*m/A)
J = 0.02    # Momento de inercia (kg*m^2)
B = 0.1     # Coeficiente de fricción

# Parámetros de simulación
h = 0.01                     # Paso de tiempo (segundos)
t = np.arange(0, 5, h)      # Vector de tiempo (0 a 5 segundos)
V = 12                      # Voltaje aplicado (V)

# Inicialización de variables
i = np.zeros(len(t))  # Corriente en el motor
w = np.zeros(len(t))  # Velocidad angular

# Método de Euler
for k in range(len(t)-1):
    
    # Ecuación de la corriente (modelo eléctrico)
    # di/dt = (1/L)*(V - R*i - Ke*w)
    i[k+1] = i[k] + h*((-R/L)*i[k] - (Ke/L)*w[k] + (1/L)*V)
    
    # Ecuación de la velocidad angular (modelo mecánico)
    # dω/dt = (1/J)*(Kt*i - B*w)
    w[k+1] = w[k] + h*((Kt/J)*i[k] - (B/J)*w[k])

# Gráfica de la corriente
plt.plot(t, i)
plt.title("Corriente vs Tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Corriente (A)")
plt.grid()

plt.show()

# Gráfica de la velocidad angular
plt.plot(t, w)
plt.title("Velocidad angular vs Tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (rad/s)")
plt.grid()

plt.show()
