"""
SEMANA 2 - Transformada de Laplace y Control PID
Motor DC - Robot Seguidor de Línea
Escuela Militar de Ingeniería - Ecuaciones Diferenciales
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# PARÁMETROS DEL MOTOR DC (motor de 3.6V - Semana 1)
# ============================================================
R  = 2.0       # Resistencia del bobinado [Ω]
L  = 0.0005    # Inductancia del bobinado [H]
Kt = 0.02      # Constante de torque [Nm/A]
Ke = 0.02      # Constante FEM [V·s/rad]
J  = 0.00001   # Momento de inercia [kg·m²]
B  = 0.0001    # Fricción viscosa [N·m·s/rad]
V  = 3.6       # Voltaje de entrada [V]

# ============================================================
# FUNCIÓN DE TRANSFERENCIA G(s) = Ω(s)/V(s)
#
# Del modelo en espacio de estados:
#   L·di/dt = V - R·i - Ke·ω   →   en Laplace: (Ls + R)·I(s) = V(s) - Ke·Ω(s)
#   J·dω/dt = Kt·i - B·ω       →   en Laplace: (Js + B)·Ω(s) = Kt·I(s)
#
# Despejando I(s) de la 2da: I(s) = (Js + B)/Kt · Ω(s)
# Sustituyendo en la 1ra:
#   (Ls + R)·(Js + B)/Kt · Ω(s) + Ke·Ω(s) = V(s)
#
# G(s) = Kt / [(Ls+R)(Js+B) + Kt·Ke]
# Numerador: Kt
# Denominador: LJ·s² + (LB + RJ)·s + (RB + Kt·Ke)
# ============================================================
num_K = Kt
den_a2 = L * J
den_a1 = L * B + R * J
den_a0 = R * B + Kt * Ke

print("=" * 60)
print("FUNCIÓN DE TRANSFERENCIA G(s) = Ω(s)/V(s)")
print("=" * 60)
print(f"\n  G(s) = {num_K}")
print(f"         " + "-" * 40)
print(f"         {den_a2:.6f}·s² + {den_a1:.6f}·s + {den_a0:.6f}")
print(f"\nCoeficientes:")
print(f"  Numerador: Kt = {num_K}")
print(f"  a2 = L·J          = {den_a2:.6e}")
print(f"  a1 = L·B + R·J    = {den_a1:.6e}")
print(f"  a0 = R·B + Kt·Ke  = {den_a0:.6e}")

# Ganancia estática (s=0): G(0) = Kt / (RB + Kt·Ke)
G0 = num_K / den_a0
print(f"\nGanancia estática G(0) = {G0:.4f} rad/s por Volt")
print(f"Velocidad en estado estable = G(0) * V = {G0 * V:.4f} rad/s")

# ============================================================
# SINTONIZACIÓN PID - MÉTODO DE ZIEGLER-NICHOLS
# 
# Para un sistema de 2do orden: usamos la aproximación del
# método de la curva de reacción (lazo abierto).
# Tiempo de retardo L_zn ~ 0 (sistema sin retardo puro),
# usamos polos del sistema para calcular Ku y Tu.
#
# Polos de la función de transferencia (raíces del denominador)
# s = [-a1 ± sqrt(a1²-4·a2·a0)] / (2·a2)
# ============================================================
discriminant = den_a1**2 - 4*den_a2*den_a0
if discriminant >= 0:
    s1 = (-den_a1 + np.sqrt(discriminant)) / (2*den_a2)
    s2 = (-den_a1 - np.sqrt(discriminant)) / (2*den_a2)
    print(f"\nPolos de G(s): s1 = {s1:.2f},  s2 = {s2:.2f}  (reales)")
else:
    real_p = -den_a1 / (2*den_a2)
    imag_p = np.sqrt(-discriminant) / (2*den_a2)
    print(f"\nPolos de G(s): s = {real_p:.2f} ± {imag_p:.2f}j  (complejos conjugados)")

# Ganancia última Ku: la ganancia proporcional que lleva el sistema
# al límite de estabilidad (parte imaginaria del denominador = 0)
# Para G(s) con controlador P: 1 + Kp·G(jω) = 0
# |den_a0 - den_a2·ωu²| = 0  →  ωu = sqrt(a0/a2)
omega_u = np.sqrt(den_a0 / den_a2)
Tu = 2 * np.pi / omega_u
# Ku = a1·ωu / Kt
Ku = (den_a1 * omega_u) / num_K

print(f"\n{'='*60}")
print("SINTONIZACIÓN ZIEGLER-NICHOLS")
print(f"{'='*60}")
print(f"  ωu (frecuencia última) = {omega_u:.4f} rad/s")
print(f"  Tu (período último)    = {Tu:.6f} s")
print(f"  Ku (ganancia última)   = {Ku:.4f}")

# Tabla Ziegler-Nichols
Kp_P   = 0.5  * Ku
Kp_PI  = 0.45 * Ku;  Ki_PI  = Kp_PI / (0.833 * Tu)
Kp_PID = 0.6  * Ku;  Ki_PID = Kp_PID / (0.5 * Tu);  Kd_PID = Kp_PID * 0.125 * Tu

print(f"\n  Tabla de parámetros Ziegler-Nichols:")
print(f"  {'Controlador':<12} {'Kp':>10} {'Ki':>12} {'Kd':>12}")
print(f"  {'-'*48}")
print(f"  {'P':<12} {Kp_P:>10.4f} {'---':>12} {'---':>12}")
print(f"  {'PI':<12} {Kp_PI:>10.4f} {Ki_PI:>12.4f} {'---':>12}")
print(f"  {'PID':<12} {Kp_PID:>10.4f} {Ki_PID:>12.4f} {Kd_PID:>12.6f}")

# ============================================================
# SIMULACIÓN DE LAZO CERRADO - Respuesta ante escalón
# Sistema: G(s) con controlador C(s) en lazo cerrado
# Gc(s) = C(s)·G(s) / (1 + C(s)·G(s))
#
# Se simula numéricamente usando método de Euler en el dominio
# del tiempo con el modelo de espacio de estados aumentado
# (incluye integrador para ki y derivador para kd)
# ============================================================
dt = 0.0001    # paso de tiempo [s]
t_end = 0.5    # tiempo total [s]
t = np.arange(0, t_end, dt)
N = len(t)
omega_ref = 100.0  # referencia de velocidad [rad/s] - escalón unitario escalado

def simulate_pid(Kp, Ki, Kd, label):
    """Simula el sistema con controlador PID en lazo cerrado."""
    i_motor = 0.0   # corriente
    omega   = 0.0   # velocidad angular
    integral_e = 0.0
    prev_e     = 0.0

    omega_hist = np.zeros(N)
    i_hist     = np.zeros(N)

    for k in range(N):
        # Error
        e = omega_ref - omega

        # PID
        integral_e += e * dt
        deriv_e = (e - prev_e) / dt if k > 0 else 0.0
        u = Kp * e + Ki * integral_e + Kd * deriv_e
        # Saturación del voltaje (realista)
        u = np.clip(u, 0.0, 12.0)

        # Dinámica del motor (Euler)
        di = (1/L) * (u - R*i_motor - Ke*omega)
        dw = (1/J) * (Kt*i_motor - B*omega)
        i_motor += di * dt
        omega   += dw * dt

        omega_hist[k] = omega
        i_hist[k]     = i_motor
        prev_e = e

    return omega_hist, i_hist

# Simulación de los tres controladores
omega_P,   i_P   = simulate_pid(Kp_P,   0,      0,       "P")
omega_PI,  i_PI  = simulate_pid(Kp_PI,  Ki_PI,  0,       "PI")
omega_PID, i_PID = simulate_pid(Kp_PID, Ki_PID, Kd_PID,  "PID")

# ============================================================
# MÉTRICAS DE DESEMPEÑO
# ============================================================
def get_metrics(omega_hist, ref):
    """Calcula overshoot, tiempo de subida y error en estado estable."""
    max_val = np.max(omega_hist)
    overshoot = (max_val - ref) / ref * 100 if max_val > ref else 0.0
    # Tiempo de subida: de 10% a 90% de la referencia
    try:
        t10 = t[np.where(omega_hist >= 0.1 * ref)[0][0]]
        t90 = t[np.where(omega_hist >= 0.9 * ref)[0][0]]
        rise_time = t90 - t10
    except IndexError:
        rise_time = float('nan')
    ess = abs(ref - omega_hist[-1]) / ref * 100
    return overshoot, rise_time, ess

os_P,   tr_P,   ess_P   = get_metrics(omega_P,   omega_ref)
os_PI,  tr_PI,  ess_PI  = get_metrics(omega_PI,  omega_ref)
os_PID, tr_PID, ess_PID = get_metrics(omega_PID, omega_ref)

print(f"\n{'='*60}")
print("MÉTRICAS DE DESEMPEÑO (referencia = 100 rad/s)")
print(f"{'='*60}")
print(f"  {'Control':<8} {'Overshoot':>12} {'t_subida (s)':>14} {'ess (%)':>10}")
print(f"  {'-'*48}")
print(f"  {'P':<8} {os_P:>11.2f}% {tr_P:>14.4f} {ess_P:>10.2f}%")
print(f"  {'PI':<8} {os_PI:>11.2f}% {tr_PI:>14.4f} {ess_PI:>10.2f}%")
print(f"  {'PID':<8} {os_PID:>11.2f}% {tr_PID:>14.4f} {ess_PID:>10.2f}%")

# ============================================================
# GRÁFICAS
# ============================================================
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#1a1a2e')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

colors = {'P': '#e94560', 'PI': '#f5a623', 'PID': '#00d4aa', 'ref': '#ffffff'}

# --- Gráfica 1: Respuesta de velocidad (todos los controladores) ---
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#16213e')
ax1.plot(t, omega_P,   color=colors['P'],   lw=2,   label=f'P   (OS={os_P:.1f}%, ess={ess_P:.1f}%)')
ax1.plot(t, omega_PI,  color=colors['PI'],  lw=2,   label=f'PI  (OS={os_PI:.1f}%, ess={ess_PI:.1f}%)')
ax1.plot(t, omega_PID, color=colors['PID'], lw=2,   label=f'PID (OS={os_PID:.1f}%, ess={ess_PID:.1f}%)')
ax1.axhline(omega_ref, color=colors['ref'], lw=1.5, ls='--', label='Referencia (100 rad/s)')
ax1.set_title('Respuesta ante Escalón – Comparación P vs PI vs PID', color='white', fontsize=13, pad=10)
ax1.set_xlabel('Tiempo [s]', color='white')
ax1.set_ylabel('Velocidad Angular ω [rad/s]', color='white')
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#444')
ax1.legend(facecolor='#16213e', labelcolor='white', fontsize=10)
ax1.grid(True, alpha=0.2)

# --- Gráfica 2: Zoom en transitorio (0 a 0.1 s) ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#16213e')
mask = t <= 0.1
ax2.plot(t[mask], omega_P[mask],   color=colors['P'],   lw=2, label='P')
ax2.plot(t[mask], omega_PI[mask],  color=colors['PI'],  lw=2, label='PI')
ax2.plot(t[mask], omega_PID[mask], color=colors['PID'], lw=2, label='PID')
ax2.axhline(omega_ref, color=colors['ref'], lw=1.5, ls='--')
ax2.set_title('Zoom: Respuesta Transitoria (0–0.1 s)', color='white', fontsize=11)
ax2.set_xlabel('Tiempo [s]', color='white')
ax2.set_ylabel('ω [rad/s]', color='white')
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#444')
ax2.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
ax2.grid(True, alpha=0.2)

# --- Gráfica 3: Error en estado estacionario ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#16213e')
ctrl_names = ['P', 'PI', 'PID']
ess_vals   = [ess_P, ess_PI, ess_PID]
bar_colors = [colors['P'], colors['PI'], colors['PID']]
bars = ax3.bar(ctrl_names, ess_vals, color=bar_colors, width=0.5)
for bar, val in zip(bars, ess_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center', va='bottom', color='white', fontsize=11)
ax3.set_title('Error en Estado Estacionario (ess)', color='white', fontsize=11)
ax3.set_xlabel('Controlador', color='white')
ax3.set_ylabel('ess [%]', color='white')
ax3.tick_params(colors='white')
ax3.spines[:].set_color('#444')
ax3.grid(True, alpha=0.2, axis='y')

plt.suptitle('Semana 2 – Control PID del Motor DC (3.6V)\nEMI – Ecuaciones Diferenciales',
             color='white', fontsize=14, y=1.01)

plt.savefig('/home/claude/semana2_graficas.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("\n✓ Gráfica guardada: semana2_graficas.png")
print("\n✓ Simulación completada exitosamente.")
