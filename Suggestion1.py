# DHA Production Prediction & Optimization with ANN + GA + Kinetic Models (Optimized with Genetic Algorithm)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import pygad

# -------------------------
# STEP 1: Simulate Time-Series Kinetic Data
# -------------------------
t = np.linspace(0, 120, 25)
X0 = 5
Xm_true = 75
mum_true = 0.08

def logistic_model(t, Xm, mum):
    return Xm / (1 + ((Xm - X0) / X0) * np.exp(-mum * t))

X_data = logistic_model(t, Xm_true, mum_true) + np.random.normal(0, 1, len(t))
popt_log, _ = curve_fit(logistic_model, t, X_data, bounds=([50, 0.01], [100, 0.15]))
Xm_fit, mum_fit = popt_log
X_fit = logistic_model(t, Xm_fit, mum_fit)

def dXdt(t, Xm, mum):
    X_t = logistic_model(t, Xm, mum)
    return mum * X_t * (1 - X_t / Xm)

dX = dXdt(t, Xm_fit, mum_fit)
X_t = logistic_model(t, Xm_fit, mum_fit)
alpha_true, beta_true = 0.12, 0.005
dPdt = alpha_true * dX + beta_true * X_t
P_data = np.cumsum(dPdt * (t[1] - t[0])) + np.random.normal(0, 0.5, len(t))

def luedeking_piret(t, alpha, beta):
    dX_sim = dXdt(t, Xm_fit, mum_fit)
    X_sim = logistic_model(t, Xm_fit, mum_fit)
    dP_sim = alpha * dX_sim + beta * X_sim
    return np.cumsum(dP_sim * (t[1] - t[0]))

popt_lp, _ = curve_fit(luedeking_piret, t, P_data, bounds=([0, 0], [1, 0.1]))
alpha_fit, beta_fit = popt_lp
P_fit = luedeking_piret(t, alpha_fit, beta_fit)

# -------------------------
# STEP 2: Generate ANN Dataset from Kinetics
# -------------------------
all_batches = []
for batch in range(5):
    df = pd.DataFrame({
        'Stage1_RPM': np.random.randint(300, 1001, size=len(t)),
        'Stage2_RPM': np.random.randint(300, 1001, size=len(t)),
        'Stage3_RPM': np.random.randint(300, 1001, size=len(t)),
        'Substrate_g_L': np.random.uniform(20, 100, size=len(t)),
        'Oxygen_percent': np.random.uniform(30, 80, size=len(t)),
        'pH': np.random.uniform(5.5, 7.5, size=len(t)),
        'Conductivity_mS': np.random.uniform(5, 25, size=len(t)),
        'DCW_g_L': X_fit + np.random.normal(0, 1, len(t)),
        'DHA_g_L': P_fit + np.random.normal(0, 0.5, len(t)),
        'Batch': batch+1
    })
    all_batches.append(df)

data = pd.concat(all_batches, ignore_index=True)

# -------------------------
# STEP 3: Export to Excel
# -------------------------

# ---------------------------------




# -------------------------
# STEP 4: Train ANN
# -------------------------
X = data[['Stage1_RPM', 'Stage2_RPM', 'Stage3_RPM', 'Substrate_g_L', 'Oxygen_percent', 'pH', 'Conductivity_mS']]
y = data[['DHA_g_L']]

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

model = MLPRegressor(hidden_layer_sizes=(15,), max_iter=1500, random_state=42)
model.fit(X_scaled, y_scaled.ravel())

# -------------------------
# STEP 5: Genetic Algorithm for Optimization
# -------------------------
def fitness_func(ga_instance, solution, solution_idx):
    input_array = np.array(solution).reshape(1, -1)
    input_df = pd.DataFrame([solution], columns=X.columns)
    input_scaled = x_scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    dha = y_scaler.inverse_transform(prediction.reshape(-1, 1))[0, 0]
    return dha

# Variable bounds: [s1, s2, s3, substrate, DO, pH, cond]
gene_space = [
    {'low': 300, 'high': 1000},  # s1
    {'low': 300, 'high': 1000},  # s2
    {'low': 300, 'high': 1000},  # s3
    {'low': 20, 'high': 100},    # substrate
    {'low': 30, 'high': 80},     # oxygen % DO
    {'low': 5.5, 'high': 7.5},   # pH
    {'low': 5, 'high': 25}       # conductivity
]

ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=7,
    gene_space=gene_space,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=20
)

ga_instance.run()
best_solution, best_fitness, _ = ga_instance.best_solution()

print("\nBest Parameters for Max DHA Yield (via GA):")
print(f"Stage1_RPM: {best_solution[0]:.0f}, Stage2_RPM: {best_solution[1]:.0f}, Stage3_RPM: {best_solution[2]:.0f}")
print(f"Substrate: {best_solution[3]:.1f} g/L, O2 (in broth): {best_solution[4]:.1f}%, pH: {best_solution[5]:.2f}, Conductivity: {best_solution[6]:.1f} mS")
print(f"Predicted Max DHA Yield: {best_fitness:.2f} g/L")

# -------------------------
# STEP 6: Plot Kinetics
# -------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, X_data, 'bo', label='Measured DCW')
plt.plot(t, X_fit, 'r-', label='Logistic Fit')
plt.xlabel('Time (h)')
plt.ylabel('DCW (g/L)')
plt.title('Biomass Growth - Logistic')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, P_data, 'go', label='Measured DHA')
plt.plot(t, P_fit, 'm-', label='Luedeking–Piret Fit')
plt.xlabel('Time (h)')
plt.ylabel('DHA (g/L)')
plt.title('Product Formation - Luedeking–Piret')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Fitted Logistic Parameters: Xm = {Xm_fit:.2f}, mum = {mum_fit:.4f}")
print(f"Fitted Luedeking-Piret Parameters: alpha = {alpha_fit:.3f}, beta = {beta_fit:.4f}")

