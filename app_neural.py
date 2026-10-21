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
        "subtitle": "Train a multi-layer neural network using Period, Plan, and Seasonality to forecast future demand accurately.",
        "about_nn_title": "💡 How does this Neural Network work? (Explained Simply)",
        "about_nn_text": """
        Imagine a team of junior analysts (neurons) working together:
        * **Input Layer:** They look at the `period` (seasonality/cycle), the `plan` (management target for the period), and past history.
        * **Hidden Layers:** They analyze non-linear patterns (e.g., *“When the plan is X during month Y, how does actual demand deviate?”*).
        * **Output Layer:** They generate the predicted `demand` for future rows where demand was left blank, **without modifying your plan**.
        """,
        "file_req": "📂 Dataset Structure & How to set up Future Periods for Forecasting",
        "file_desc": """
        Ensure your dataset has:
        * **`period`** (e.g., 1 to 12, or continuous numbers. The app handles seasonality automatically).
        * **`plan`** (Management target/budget — **must be filled for ALL rows, including future periods**).
        * **`demand`** (Actual sales — **leave blank/empty for future rows** you want to forecast!).
        """,
        "table_example_caption": "📋 Example of a correct input table structure:",
        "upload_label": "Upload your dataset (.xlsx or .csv)",
        "preview": "### Data Preview:",
        "target_col": "Select Target Column (Actual Demand / Sales):",
        "plan_col": "Select Plan Column:",
        "period_col": "Select Period Column:",
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
        "subtitle": "Навчіть багатошарову нейромережу на основі періоду, плану та сезонності для точного прогнозування.",
        "about_nn_title": "💡 Як працює ця нейромережа? (Пояснення «на пальцях»)",
        "about_nn_text": """
        Уявіть команду аналітиків (нейронів):
        * **Вхідний шар (Input):** Аналізують `період` (враховуючи річну циклічність/сезонність), `план` на цей період та історію.
        * **Приховані шари (Hidden Layers):** Шукають зв'язки (наприклад: *«якщо в такому місяці план становить X, наскільки реальний попит зазвичай відхиляється?»*).
        * **Вихідний шар (Output):** Заповнює прогнозний `demand` у тих рядках, де він був залишений порожнім, **абсолютно не чіпаючи ваш план**.
        """,
        "file_req": "📂 Структура файлу та як підготувати майбутні періоди для прогнозу",
        "file_desc": """
        Ваш файл має містити:
        * Колонку з періодом (`period` — номери місяців або циклічність)
        * Колонку з планом (`plan` — заповнена **для ВСІХ рядків, включаючи майбутні періоди**, щоб модель знала ціль)
        * Колонку з фактом (`demand` — для історії заповнена, а **для майбутніх місяців залишена порожньою**!)
        """,
        "table_example_caption": "📋 Приклад правильного оформлення таблиці:",
        "upload_label": "Завантажте ваш набір даних (.xlsx або .csv)",
        "preview": "### Попередній перегляд даних:",
        "target_col": "Оберіть цільову колонку (Фактичний попит / Demand):",
        "plan_col": "Оберіть колонку з Планом (Plan):",
        "period_col": "Оберіть колонку з Періодом (Period):",
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
selected_lang = st.sidebar.selectbox("🌐 Language / Моva", ["EN", "UA"])
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
        "period": [1, 2, 3, 4, 5, 6],
        "plan": [150, 150, 150, 150, 176, 185],
        "demand": [155, 146, 146, 146, 172, ""]
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
        period_col = st.selectbox(t["period_col"], [c for c in df.columns if c not in [target_col, plan_col]])

        if st.button(t["train_btn"], type="primary"):
            working_df = df.copy()
            
            # Конвертація в числа
            working_df[target_col] = pd.to_numeric(working_df[target_col], errors='coerce')
            working_df[plan_col] = pd.to_numeric(working_df[plan_col], errors='coerce')
            working_df[period_col] = pd.to_numeric(working_df[period_col], errors='coerce')
            
            # Додаємо циклічний місяць (сезонність від 1 до 12 по кругу)
            working_df['Cyclic_Month'] = ((working_df[period_col] - 1) % 12) + 1
            working_df['Row_Index'] = range(len(working_df))
            
            # Історичні рядки: де є і plan, і demand, і period
            train_df = working_df.dropna(subset=[target_col, plan_col, period_col]).copy()
            
            if len(train_df) < 3:
                st.error("⚠️ Error: Not enough valid historical data rows where Plan, Demand, and Period are filled. Please check your data.")
            else:
                # Вхідні параметри для нейромережі: циклічний місяць + план
                X_train = train_df[['Cyclic_Month', plan_col]].values
                y_train = train_df[target_col].values
                
                # Масштабування
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train_scaled = scaler_x.fit_transform(X_train)
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
                
                try:
                    hidden_layer_sizes = tuple(map(int, layers_input.split(',')))
                except:
                    hidden_layer_sizes = (50, 50)
                
                # Навчання нейромережі
                mlp = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    random_state=42
                )
                
                mlp.fit(X_train_scaled, y_train_scaled)
                
                # Прогноз для всіх рядків, де заповнений план і період
                valid_mask = working_df[plan_col].notna() & working_df[period_col].notna()
                X_all = working_df.loc[valid_mask, ['Cyclic_Month', plan_col]].values
                
                X_all_scaled = scaler_x.transform(X_all)
                y_all_pred_scaled = mlp.predict(X_all_scaled)
                
                working_df['Neural_Net_Forecast'] = np.nan
                working_df.loc[valid_mask, 'Neural_Net_Forecast'] = np.round(
                    scaler_y.inverse_transform(y_all_pred_scaled.reshape(-1, 1)).ravel(), 2
                )
                
                # Фінальна колонка: історія залишається фактом, майбутнє заповнюється прогнозом нейромережі
                working_df['Final_Output'] = working_df[target_col]
                missing_indices = working_df[target_col].isna()
                working_df.loc[missing_indices, 'Final_Output'] = working_df.loc[missing_indices, 'Neural_Net_Forecast']

                st.success(t["success_msg"])
                st.dataframe(working_df)

                # Візуалізація
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(working_df['Row_Index'], working_df[target_col], label="Historical Demand (Actual)", color="black", marker="o")
                ax.plot(working_df['Row_Index'], working_df['Neural_Net_Forecast'], label="Neural Network Forecast", color="purple", linestyle="--", marker="x")
                ax.plot(working_df['Row_Index'], working_df[plan_col], label="Plan (Untouched)", color="orange", linestyle=":")

                ax.set_title(t["chart_title"])
                ax.legend()
                ax.grid(True, linestyle=":", alpha=0.6)
                st.pyplot(fig)

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
