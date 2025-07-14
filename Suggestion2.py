import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pygad
import io
from scipy.optimize import curve_fit

# ---------- Simulate Kinetic Model (Logistic + Luedeking-Piret) ----------
t = np.linspace(0, 120, 25)
X0 = 5
Xm_true, mum_true = 75, 0.08

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

# ---------- Create ANN Dataset ----------
feature_columns = ['Stage1_RPM', 'Stage2_RPM', 'Stage3_RPM', 'Substrate_g_L', 'Oxygen_percent', 'pH', 'Conductivity_mS']
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
        'Batch': batch + 1
    })
    all_batches.append(df)

data = pd.concat(all_batches, ignore_index=True)
X = data[feature_columns]
y = data[['DHA_g_L']]

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

model = MLPRegressor(hidden_layer_sizes=(15,), max_iter=1500, random_state=42)
model.fit(X_scaled, y_scaled.ravel())

# ---------- GA Optimization ----------
gene_space = [
    {'low': 300, 'high': 1000},
    {'low': 300, 'high': 1000},
    {'low': 300, 'high': 1000},
    {'low': 20, 'high': 100},
    {'low': 30, 'high': 80},
    {'low': 5.5, 'high': 7.5},
    {'low': 5, 'high': 25}
]
gene_space_dict = dict(zip(feature_columns, gene_space))

def fitness_func(ga_instance, solution, solution_idx):
    input_df = pd.DataFrame([solution], columns=feature_columns)
    input_scaled = x_scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    dha = y_scaler.inverse_transform(prediction.reshape(-1, 1))[0, 0]
    return dha

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
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    ga_instance.run()

best_solution, best_fitness, _ = ga_instance.best_solution()

# ---------- Streamlit UI ----------
st.title("DHA Optimization using ANN + GA + Kinetics")

st.subheader("Best Parameters for Max DHA Yield (via GA):")
st.write(dict(zip(feature_columns, best_solution)))
st.success(f"Predicted Max DHA Yield: {best_fitness:.2f} g/L")

# ---------- Manual Testing ----------
st.subheader("Manual Input Testing")
manual_input = {}
for col, gs in zip(feature_columns, gene_space):
    manual_input[col] = st.slider(col, float(gs["low"]), float(gs["high"]), float((gs["low"] + gs["high"]) / 2))

input_df = pd.DataFrame([manual_input])
input_scaled = x_scaler.transform(input_df)
manual_pred = model.predict(input_scaled)
manual_dha = y_scaler.inverse_transform(manual_pred.reshape(-1, 1))[0, 0]
st.info(f"Predicted DHA for Manual Input: {manual_dha:.2f} g/L")

# ---------- Yield Curves ----------
st.subheader("Yield Curves")
selected_variable = st.selectbox("Select Variable for Yield Curve", feature_columns)
base_input = dict(zip(feature_columns, best_solution))

def plot_yield_curve(variable):
    x_vals = np.linspace(gene_space_dict[variable]['low'], gene_space_dict[variable]['high'], 100)
    predictions = []
    for val in x_vals:
        test_input = base_input.copy()
        test_input[variable] = val
        df = pd.DataFrame([test_input], columns=feature_columns)
        scaled = x_scaler.transform(df)
        pred = model.predict(scaled)
        dha = y_scaler.inverse_transform(pred.reshape(-1, 1))[0, 0]
        predictions.append(dha)

    fig, ax = plt.subplots()
    ax.plot(x_vals, predictions)
    ax.set_xlabel(variable)
    ax.set_ylabel("Predicted DHA (g/L)")
    ax.set_title(f"Yield Curve - {variable}")
    st.pyplot(fig)

plot_yield_curve(selected_variable)

# ---------- GA Convergence ----------
st.subheader("GA Convergence Plot")
fig2, ax2 = plt.subplots()
ax2.plot(ga_instance.best_solutions_fitness)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Best DHA Yield")
ax2.set_title("GA Convergence")
st.pyplot(fig2)

# ---------- Upload Sensor Data ----------
st.subheader("Live Fermentation Sensor Data")
uploaded_file = st.file_uploader("Upload Live Sensor CSV", type=["csv"])
if uploaded_file:
    live_data = pd.read_csv(uploaded_file)
    st.write("Live Data Preview:", live_data.head())

# ---------- Download Section ----------
st.subheader("Download Optimized Result")
result_df = pd.DataFrame([best_solution], columns=feature_columns)
result_df['Predicted_DHA'] = best_fitness
st.download_button("Download Results as CSV", result_df.to_csv(index=False), file_name='DHA_results.csv')
