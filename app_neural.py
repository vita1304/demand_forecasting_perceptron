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
        "subtitle": "Train a neural network on historical Plan & Seasonality to forecast future Demand without modifying your Plan.",
        "about_nn_title": "💡 How does this Neural Network work? (Explained Simply)",
        "about_nn_text": """
        * **Training:** The neural network learns from historical `period` (seasonality/cycle) and `plan` to predict `demand`.
        * **Future Forecasting:** For future periods where `demand` is blank, the model forecasts the demand. 
        * **Plan Protection:** Your original `plan` column is **never touched or overwritten**! If future plans are blank, the model estimates them internally for calculation only.
        """,
        "file_req": "📂 Dataset Structure & How to set up Future Periods",
        "file_desc": """
        Ensure your dataset has 3 main columns:
        * **`period`** (e.g. 1, 2, 3... 12, 13, 14...)
        * **`plan`** (Management target)
        * **`demand`** (Actual sales — **leave blank/empty for future rows** you want to forecast!)
        """,
        "table_example_caption": "📋 Example table structure:",
        "upload_label": "Upload your dataset (.xlsx or .csv)",
        "preview": "### Data Preview:",
        "target_col": "Select Target Column (Demand / Actual):",
        "plan_col": "Select Plan Column:",
        "period_col": "Select Period Column:",
        "params_title": "⚙️ Neural Network Hyperparameters",
        "hidden_layers": "Hidden Layers Structure:",
        "hl_help": "E.g., `50, 50` means two hidden layers with 50 neurons each.",
        "learning_rate": "Initial Learning Rate:",
        "lr_help": "Learning rate (default: 0.001).",
        "max_iter": "Max Training Iterations (Epochs):",
        "mi_help": "Number of training epochs.",
        "train_btn": "🚀 Train Neural Network & Predict Demand",
        "success_msg": "✅ Neural network successfully trained, future demand forecasted!",
        "chart_title": "Neural Network Forecast vs History",
        "download_btn": "📥 Download Results as Excel"
    },
    "UA": {
        "title": "🧠 Нейромережевий хаб прогнозування попиту та продажів (MLP)",
        "subtitle": "Навчіть нейромережу на історичних даних План і Сезонність для точного прогнозу попиту, не чіпаючи ваш План.",
        "about_nn_title": "💡 Як працює ця нейромережа? (Пояснення «на пальцях»)",
        "about_nn_text": """
        * **Навчання:** Мережа вивчає зв'язок між `періодом` (сезонністю/циклом), `планом` та фактичним `demand` у минулому.
        * **Прогноз:** Для майбутніх місяців, де `demand` залишено порожнім, модель розраховує прогнозний попит.
        * **Захист плану:** Ваша колонка `plan` **ніколи не перезаписується і не чіпається**! Якщо в майбутніх рядках план порожній, модель використовує внутрішню оцінку для розрахунку, залишаючи ваші вхідні дані в безпеці.
        """,
        "file_req": "📂 Структура файлу та підготовка майбутніх періодів",
        "file_desc": """
        Ваш файл має містити 3 колонки:
        * **`period`** (номери періодів/місяців: 1, 2, 3...)
        * **`plan`** (планові показники)
        * **`demand`** (фактичний попит — **для майбутніх рядків залиште цю клітинку абсолютно порожньою**!)
        """,
        "table_example_caption": "📋 Приклад правильної структури таблиці:",
        "upload_label": "Завантажте ваш набір даних (.xlsx або .csv)",
        "preview": "### Попередній перегляд даних:",
        "target_col": "Оберіть цільову колонку (Фактичний попит / Demand):",
        "plan_col": "Оберіть колонку з Планом (Plan):",
        "period_col": "Оберіть колонку з Періодом (Period):",
        "params_title": "⚙️ Налаштування нейромережі",
        "hidden_layers": "Структура прихованих шарів:",
        "hl_help": "Наприклад, `50, 50` — два шари по 50 нейронів.",
        "learning_rate": "Швидкість навчання (Learning Rate):",
        "lr_help": "Швидкість адаптації ваг (рекомендовано 0.001).",
        "max_iter": "Максимум епох навчання (Iterations):",
        "mi_help": "Кількість циклів навчання мережі.",
        "train_btn": "🚀 Навчити нейромережу та спрогнозувати Demand",
        "success_msg": "✅ Нейромережу успішно навчено, прогноз попиту сформовано!",
        "chart_title": "Прогноз нейромережі у порівнянні з історією",
        "download_btn": "📥 Завантажити результати в Excel"
    }
}

selected_lang = st.sidebar.selectbox("🌐 Language / Мова", ["EN", "UA"])
t = translations[selected_lang]

st.title(t["title"])
st.markdown(t["subtitle"])

with st.expander(t["about_nn_title"], expanded=False):
    st.markdown(t["about_nn_text"])

st.sidebar.markdown(f"### {t['params_title']}")
layers_input = st.sidebar.text_input(t["hidden_layers"], value="50, 50")
st.sidebar.caption(t["hl_help"])

learning_rate_init = st.sidebar.number_input(t["learning_rate"], min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
st.sidebar.caption(t["lr_help"])

max_iter = st.sidebar.slider(t["max_iter"], min_value=100, max_value=2000, value=500, step=100)
st.sidebar.caption(t["mi_help"])

with st.expander(t["file_req"], expanded=True):
    st.markdown(t["file_desc"])
    st.markdown(t["table_example_caption"])
    
    sample_df = pd.DataFrame({
        "period": [1, 2, 3, 4, 5, 6],
        "plan": [150, 150, 150, 150, 176, ""],
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

        cols = list(df.columns)
        target_col = st.selectbox(t["target_col"], cols)
        plan_col = st.selectbox(t["plan_col"], [c for c in cols if c != target_col])
        period_col = st.selectbox(t["period_col"], [c for c in cols if c not in [target_col, plan_col]])

        if st.button(t["train_btn"], type="primary"):
            working_df = df.copy()
            
            # Конвертація в числа
            working_df[target_col] = pd.to_numeric(working_df[target_col], errors='coerce')
            working_df[plan_col] = pd.to_numeric(working_df[plan_col], errors='coerce')
            working_df[period_col] = pd.to_numeric(working_df[period_col], errors='coerce')
            
            # Розрахунок циклічного місяця для сезонності (1-12 по кругу)
            working_df['Cyclic_Month'] = ((working_df[period_col] - 1) % 12) + 1
            working_df['Row_Index'] = range(len(working_df))
            
            # Навчальна вибірка: де є факт (demand), план і період
            train_df = working_df.dropna(subset=[target_col, plan_col, period_col]).copy()
            
            if len(train_df) < 3:
                st.error("⚠️ Error: Not enough valid historical rows where Demand, Plan, and Period are filled.")
            else:
                # Входи для моделі: циклічний місяць + план
                X_train = train_df[['Cyclic_Month', plan_col]].values
                y_train = train_df[target_col].values
                
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train_scaled = scaler_x.fit_transform(X_train)
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
                
                try:
                    hidden_layer_sizes = tuple(map(int, layers_input.split(',')))
                except:
                    hidden_layer_sizes = (50, 50)
                
                mlp = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    random_state=42
                )
                
                mlp.fit(X_train_scaled, y_train_scaled)
                
                # Підготовка даних для прогнозу всіх рядків
                # Якщо план відсутній у майбутньому, підставляємо середнє для обчислення моделі, НЕ чіпаючи оригінальну колонку plan
                calc_plan = working_df[plan_col].copy()
                mean_plan = train_df[plan_col].mean()
                calc_plan.fillna(mean_plan, inplace=True)
                
                X_all = working_df[['Cyclic_Month']].copy()
                X_all['Plan_Calc'] = calc_plan
                X_all_vals = X_all.values
                
                X_all_scaled = scaler_x.transform(X_all_vals)
                y_all_pred_scaled = mlp.predict(X_all_scaled)
                
                working_df['Neural_Net_Forecast'] = np.round(
                    scaler_y.inverse_transform(y_all_pred_scaled.reshape(-1, 1)).ravel(), 2
                )
                
                # Фінальна колонка: для історії залишається реальний demand, для майбутнього вставляємо прогноз нейромережі
                working_df['Final_Output'] = working_df[target_col]
                missing_indices = working_df[target_col].isna()
                working_df.loc[missing_indices, 'Final_Output'] = working_df.loc[missing_indices, 'Neural_Net_Forecast']

                st.success(t["success_msg"])
                st.dataframe(working_df)

                # Графік
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
