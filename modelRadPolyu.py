import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
M0 = 5.0       # mol/L, initial monomer
I0 = 0.05      # mol/L, initial initiator
f = 0.6        # initiator efficiency
kd = 2.5e-5    # 1/s, decomposition rate constant
kp = 341       # L/mol/s, propagation rate
kt = 1.5e7     # L/mol/s, termination rate
MW_m = 100.12  # g/mol, MMA

# Radical concentration (pseudo-steady state)
R_rad = np.sqrt(f * kd * I0 / kt)

# === Time Setup ===
t_end = 20000  # seconds
dt = 1
time = np.arange(0, t_end + dt, dt)

# === Initialize Arrays ===
M = np.zeros_like(time)  # monomer
X = np.zeros_like(time)  # conversion
Mn = np.zeros_like(time) # molecular weight

M[0] = M0

# === Simulation Loop ===
for i in range(1, len(time)):
    Rp = kp * M[i-1] * R_rad
    dM = -Rp * dt
    M[i] = max(M[i-1] + dM, 0)
    X[i] = (M0 - M[i]) / M0
    Mn[i] = MW_m * X[i] * M0 / R_rad


# === Plot Results ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(time / 60, M)
plt.title('Monomer vs Time')
plt.xlabel('Time (min)')
plt.ylabel('[M] (mol/L)')

plt.subplot(1, 3, 2)
plt.plot(time / 60, X)
plt.title('Conversion vs Time')
plt.xlabel('Time (min)')
plt.ylabel('Conversion')

plt.subplot(1, 3, 3)
plt.plot(time / 60, Mn / 1000)
plt.title('Mn vs Time')
plt.xlabel('Time (min)')
plt.ylabel('Mn (kg/mol)')

plt.tight_layout()
plt.show()

if np.any(M < 0):
    print("Warning: Negative monomer concentration detected.")
if np.any(np.isnan(Mn)):
    print("Warning: Mn contains NaN values.")
