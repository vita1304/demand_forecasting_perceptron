import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Імпорт нейромережі з scikit-learn (MLPRegressor)
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_MLP = True
except ImportError:
    HAS_MLP = False

st.set_page_config(
    page_title="Neural Network Demand & Sales Forecasting Hub",
    layout="wide"
)

# Двомовний інтерфейс (English за замовчуванням, UA перемикач)
translations = {
    "EN": {
        "title": "🧠 Neural Network Demand & Sales Forecasting Hub (MLP)",
        "subtitle": "Train a multi-layer neural network using Period, Plan, and historical Demand/Sales to forecast future periods accurately.",
        "about_nn_title": "💡 How does this Neural Network work? (Explained Simply)",
        "about_nn_text": """
        Imagine a team of junior analysts (neurons) working together:
        * **Input Layer:** They look at the `Period`, the `Plan` (what management planned), and past history.
        * **Hidden Layers:** They analyze non-linear patterns (e.g., *“When the plan is set to X, how does actual demand usually deviate?”*).
        * **Output Layer:** They combine their conclusions to output the accurate forecast for future periods where demand was left blank.
        """,
        "file_req": "📂 Dataset Structure & How to set up Future Periods for Forecasting",
        "file_desc": """
        For stable learning, ensure your dataset has:
        * **`period`** (or Time Index)
        * **`plan`** (Management target/budget for every row, including future periods)
        * **`demand`** (Actual sales/consumption — **leave this blank/empty for future periods** you want to predict!)
        """,
        "table_example_caption": "📋 Example of a correct input table structure:",
        "upload_label": "Upload your dataset (.xlsx or .csv)",
        "preview": "### Data Preview:",
        "target_col": "Select Target Column (Actual Demand / Sales):",
        "plan_col": "Select Plan Column:",
        "params_title": "⚙️ Neural Network Hyperparameters Guide",
        "hidden_layers": "Hidden Layers Structure:",
        "hl_help": "Defines the architecture. E.g., `50, 50` means two hidden layers with 50 neurons each.",
        "learning_rate": "Initial Learning Rate:",
        "lr_help": "Controls how fast weights update (default: 0.001).",
        "max_iter": "Max Training Iterations (Epochs):",
        "mi_help": "How many times the network reviews the dataset during training.",
        "train_btn": "🚀 Train Neural Network & Predict",
        "success_msg": "✅ Neural network successfully trained, future demand predicted!",
        "chart_title": "Neural Network Forecast vs History",
        "download_btn": "📥 Download Results as Excel"
    },
    "UA": {
        "title": "🧠 Нейромережевий хаб прогнозування попиту та продажів (MLP)",
        "subtitle": "Навчіть багатошарову нейромережу на основі періоду, плану та фактичних даних для точного прогнозування майбутнього.",
        "about_nn_title": "💡 Як працює ця нейромережа? (Пояснення «на пальцях»)",
        "about_nn_text": """
        Уявіть команду аналітиків (нейронів):
        * **Вхідний шар (Input):** Вивчають `період`, `план` (що планувалося) та історію.
        * **Приховані шари (Hidden Layers):** Шукають закономірності (наприклад: *«якщо план становить таку цифру, наскільки реальний попит зазвичай відхиляється?»*).
        * **Вихідний шар (Output):** Заповнює порожні клітинки майбутніх періодів прогнозом.
        """,
        "file_req": "📂 Структура файлу та як підготувати майбутні періоди для прогнозу",
        "file_desc": """
        Ваш файл має містити:
        * Колонку з періодом (наприклад, `period`)
        * Колонку з планом (`plan` — заповнена і для історії, і для майбутніх періодів, які треба спрогнозувати)
        * Колонку з фактом (`demand` — для історії заповнена, а **для майбутніх місяців залишена порожньою**!)
        """,
        "table_example_caption": "📋 Приклад правильного оформлення таблиці:",
        "upload_label": "Завантажте ваш набір даних (.xlsx або .csv)",
        "preview": "### Попередній перегляд даних:",
        "target_col": "Оберіть цільову колонку (Фактичний попит / Demand):",
        "plan_col": "Оберіть колонку з Планом (Plan):",
        "params_title": "⚙️ Гід по параметрах нейромережі",
        "hidden_layers": "Структура прихованих шарів:",
        "hl_help": "Наприклад, `50, 50` — це два шари по 50 нейронів.",
        "learning_rate": "Швидкість навчання (Learning Rate):",
        "lr_help": "Швидкість виправлення помилок мережею (рекомендовано 0.001).",
        "max_iter": "Максимум епох навчання (Iterations):",
        "mi_help": "Кількість повних проходів мережі по набору даних.",
        "train_btn": "🚀 Навчити нейромережу та зробити прогноз",
        "success_msg": "✅ Нейромережу успішно навчено, прогноз попиту сформовано!",
        "chart_title": "Прогноз нейромережі у порівнянні з історією",
        "download_btn": "📥 Завантажити результати в Excel"
    }
}

# Вибір мови
selected_lang = st.sidebar.selectbox("🌐 Language / Мова", ["EN", "UA"])
t = translations[selected_lang]

st.title(t["title"])
st.markdown(t["subtitle"])

with st.expander(t["about_nn_title"], expanded=False):
    st.markdown(t["about_nn_text"])

# Налаштування гіперпараметрів у сайдбарі
st.sidebar.markdown(f"### {t['params_title']}")
layers_input = st.sidebar.text_input(t["hidden_layers"], value="50, 50")
st.sidebar.caption(t["hl_help"])

learning_rate_init = st.sidebar.number_input(t["learning_rate"], min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
st.sidebar.caption(t["lr_help"])

max_iter = st.sidebar.slider(t["max_iter"], min_value=100, max_value=2000, value=500, step=100)
st.sidebar.caption(t["mi_help"])

# Вимоги до файлу та наочна таблиця-приклад
with st.expander(t["file_req"], expanded=True):
    st.markdown(t["file_desc"])
    st.markdown(t["table_example_caption"])
    
    sample_df = pd.DataFrame({
        "period": [1, 2, 3, 4, 5],
        "plan": [150, 150, 176, 185, 200],
        "demand": [155, 146, 172, "", ""]
    })
    st.dataframe(sample_df, use_container_width=True)

uploaded_file = st.file_uploader(t["upload_label"], type=["xlsx", "csv"])

if uploaded_file is not None and HAS_MLP:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write(t["preview"])
        st.dataframe(df.head())

        target_col = st.selectbox(t["target_col"], df.columns)
        plan_col = st.selectbox(t["plan_col"], [c for c in df.columns if c != target_col])

        if st.button(t["train_btn"], type="primary"):
            working_df = df.copy()
            
            # Визначаємо історичні рядки (де є фактичний demand) та майбутні (де порожньо)
            historical_mask = working_df[target_col].notna() & (working_df[target_col] != "")
            
            train_df = working_df[historical_mask].copy()
            predict_df = working_df[~historical_mask].copy()
            
            if len(train_df) < 3:
                st.error("⚠️ Error: Not enough historical data rows with values in the target column to train the model.")
            else:
                # Вхідні параметри для навчання: індекс часу (номер рядка) та план
                train_df['Time_Index'] = range(len(train_df))
                working_df['Time_Index'] = range(len(working_df))
                
                X_train = train_df[['Time_Index', plan_col]].values
                y_train = train_df[target_col].values
                
                # Масштабування даних для стабільної роботи нейромережі
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train_scaled = scaler_x.fit_transform(X_train)
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
                
                try:
                    hidden_layer_sizes = tuple(map(int, layers_input.split(',')))
                except:
                    hidden_layer_sizes = (50, 50)
                
                # Навчання нейромережі (MLPRegressor)
                mlp = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    random_state=42
                )
                
                mlp.fit(X_train_scaled, y_train_scaled)
                
                # Робимо прогноз для ВСІХ рядків (і для історії, і для майбутнього)
                X_all = working_df[['Time_Index', plan_col]].values
                X_all_scaled = scaler_x.transform(X_all)
                y_all_pred_scaled = mlp.predict(X_all_scaled)
                
                working_df['Neural_Net_Forecast'] = np.round(
                    scaler_y.inverse_transform(y_all_pred_scaled.reshape(-1, 1)).ravel(), 2
                )
                
                # Формуємо фінальну вихідну колоночку: для історії залишаємо факт, а для майбутнього вставляємо прогноз нейромережі
                working_df['Final_Output'] = working_df[target_col]
                missing_indices = working_df[target_col].isna() | (working_df[target_col] == "")
                working_df.loc[missing_indices, 'Final_Output'] = working_df.loc[missing_indices, 'Neural_Net_Forecast']

                st.success(t["success_msg"])
                st.dataframe(working_df)

                # Візуалізація результатів на графіку
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(working_df['Time_Index'], working_df[target_col], label="Historical Demand (Actual)", color="black", marker="o")
                ax.plot(working_df['Time_Index'], working_df['Neural_Net_Forecast'], label="Neural Network Forecast", color="purple", linestyle="--", marker="x")
                ax.plot(working_df['Time_Index'], working_df[plan_col], label="Plan", color="orange", linestyle=":")

                ax.set_title(t["chart_title"])
                ax.legend()
                ax.grid(True, linestyle=":", alpha=0.6)
                st.pyplot(fig)

                # Експорт результатів у Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    working_df.to_excel(writer, index=False, sheet_name="NeuralNet_Forecast")
                
                st.download_button(
                    label=t["download_btn"],
                    data=output.getvalue(),
                    file_name="neural_net_forecast_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error during neural network training: {e}")
elif not HAS_MLP:
    st.error("Scikit-learn is not installed properly.")
