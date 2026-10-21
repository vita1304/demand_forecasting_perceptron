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
        "subtitle": "Train a multi-layer neural network on historical data (supporting Plan vs Actual features) to predict future sales or demand.",
        "about_nn_title": "💡 How does this Neural Network work? (Explained Simply)",
        "about_nn_text": """
        Imagine a team of junior analysts (neurons) working together:
        * **Input Layer:** They look at historical data (e.g., past demand/sales, planned targets, or time index).
        * **Hidden Layers:** They talk to each other, testing combinations and non-linear patterns (e.g., *“If the planned volume was high, but actual sales dropped last month, how will it affect next month?”*).
        * **Output Layer:** They combine their conclusions to output the final forecast for future periods.
        
        *Note: The exact same logic and mechanics apply whether you are forecasting **Demand**, **Sales**, or any other supply chain metric!*
        """,
        "file_req": "📂 Dataset Structure & How to set up Future Periods for Forecasting",
        "file_desc": """
        For stable learning, it is recommended to have at least **12–24 historical rows**.
        * **How to structure your file for future forecasting:**
          1. In the upper rows, write your historical data (e.g., Months 1 to 12) where you have both **Plan** and actual **Demand / Sales**.
          2. In the lower rows (future periods you want to forecast, e.g., Months 13, 14, 15), leave the target column (**Demand / Sales**) **completely empty (blank)**. 
          3. Fill in the **Plan** column for those future months if you have targets.
          4. The neural network will train on your history and automatically fill in the blank cells with its intelligent forecast!
        """,
        "table_example_caption": "📋 Example of how your input table should look (notice the blank cells for future forecast):",
        "upload_label": "Upload your dataset (.xlsx or .csv)",
        "preview": "### Data Preview:",
        "target_col": "Select Target Column (Actual Demand / Sales):",
        "feature_col": "Select Optional Feature Column (e.g., Plan / Target):",
        "use_feature": "Use Plan/Feature column for training",
        "params_title": "⚙️ Neural Network Hyperparameters Guide",
        "hidden_layers": "Hidden Layers Structure:",
        "hl_help": "Defines the architecture. E.g., `50, 50` means two hidden layers with 50 neurons each. More layers catch complex patterns, but require more data.",
        "learning_rate": "Initial Learning Rate:",
        "lr_help": "Controls how fast weights update. Too high = model jumps around and fails to learn. Too low = training takes forever (default: 0.001).",
        "max_iter": "Max Training Iterations (Epochs):",
        "mi_help": "How many times the network reviews the entire dataset during training.",
        "train_btn": "🚀 Train Neural Network & Predict",
        "success_msg": "✅ Neural network successfully trained and forecast generated for future periods!",
        "chart_title": "Neural Network Forecast vs History",
        "download_btn": "📥 Download Results as Excel"
    },
    "UA": {
        "title": "🧠 Нейромережевий хаб прогнозування попиту та продажів (MLP)",
        "subtitle": "Навчіть багатошарову нейромережу на історичних даних (з підтримкою факторів План проти Факт) для прогнозування майбутніх продажів чи потреби.",
        "about_nn_title": "💡 Як працює ця нейромережа? (Пояснення «на пальцях»)",
        "about_nn_text": """
        Уявіть команду аналітиків (нейронів):
        * **Вхідний шар (Input):** Аналітики вивчають історію (попередній попит/продажі, планові показники чи час).
        * **Приховані шари (Hidden Layers):** Вони обговорюють зв'язки (наприклад: *«якщо планові закупівлі зросли, а минулого місяця був спад, як це вплине на майбутнє?»*).
        * **Вихідний шар (Output):** Видає фінальний прогноз для майбутніх періодів.
        
        *Примітка: Ця ж сама логіка та математика працює як для прогнозування попиту (**Demand**), так і для прогнозування продажів (**Sales Forecast**) чи будь-яких інших подібних показників у бізнесі!*
        """,
        "file_req": "📂 Структура файлу та як підготувати майбутні періоди для прогнозу",
        "file_desc": """
        Для стабільного навчання бажано мати хоча б **12–24 історичні рядки**.
        * **Як правильно підготувати файл:**
          1. У верхніх рядках запишіть історію (наприклад, місяці 1–12), де є і **Plan**, і фактичний **Demand / Sales**.
          2. У нижніх рядках (періоди, які ви хочете спрогнозувати уперед, наприклад, місяці 13, 14, 15) залиште цільову колонку (**Demand / Sales**) **абсолютно порожньою**.
          3. У колонці **Plan** для цих майбутніх місяців можете вказати планові показники.
          4. Нейромережа сама навчиться на історії, а в порожні клітинки майбутніх місяців запише свій прогноз!
        """,
        "table_example_caption": "📋 Приклад того, як має виглядати ваша таблиця (зверніть увагу на порожні клітинки для прогнозу):",
        "upload_label": "Завантажте ваш набір даних (.xlsx або додатковий файл)",
        "preview": "### Попередній перегляд даних:",
        "target_col": "Оберіть цільову колонку (Фактичний попит / Продажі):",
        "feature_col": "Оберіть додатковий фактор (наприклад, План / Бюджет):",
        "use_feature": "Використовувати колонку План для навчання",
        "params_title": "⚙️ Гід по параметрах нейромережі",
        "hidden_layers": "Структура прихованих шарів:",
        "hl_help": "Визначає «розум» мережі. Наприклад, `50, 50` — це два шари по 50 нейронів. Більше шарів — складніші закономірності, але потрібно більше даних.",
        "learning_rate": "Швидкість навчання (Learning Rate):",
        "lr_help": "Швидкість, з якою нейрони виправляють помилки. Завелико — мережа «стрибає» і не вчиться; замало — вчиться надто повільно (рекомендовано 0.001).",
        "max_iter": "Максимум епох навчання (Iterations):",
        "mi_help": "Скільки разів нейромережа повністю перегляне ваш масив даних у процесі тренування.",
        "train_btn": "🚀 Навчити нейромережу та зробити прогноз",
        "success_msg": "✅ Нейромережу успішно навчено, прогноз на майбутні періоди сформовано!",
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

# Налаштування гіперпараметрів із детальними підказками у сайдбарі
st.sidebar.markdown(f"### {t['params_title']}")

layers_input = st.sidebar.text_input(t["hidden_layers"], value="50, 50")
st.sidebar.caption(t["hl_help"])

learning_rate_init = st.sidebar.number_input(t["learning_rate"], min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
st.sidebar.caption(t["lr_help"])

max_iter = st.sidebar.slider(t["max_iter"], min_value=100, max_value=2000, value=500, step=100)
st.sidebar.caption(t["mi_help"])

# Вимоги до файлу та наочна таблиця-приклад із порожніми клітинками
with st.expander(t["file_req"], expanded=True):
    st.markdown(t["file_desc"])
    st.markdown(t["table_example_caption"])
    
    sample_df = pd.DataFrame({
        "Period": ["Month 1 (History)", "Month 2 (History)", "Month 3 (History)", "Month 4 (Forecast Target)", "Month 5 (Forecast Target)"],
        "Plan": [1000, 1100, 1050, 1200, 1250],
        "Demand_or_Sales": [950, 1120, 1020, "", ""]
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
        
        use_feat = st.checkbox(t["use_feature"], value=False)
        feature_col = None
        if use_feat:
            feature_col = st.selectbox(t["feature_col"], [c for c in df.columns if c != target_col])

        if st.button(t["train_btn"], type="primary"):
            working_df = df.copy()
            working_df['Time_Index'] = range(len(working_df))
            
            # Відокремлюємо історичні рядки (де є фактичний таргет) від майбутніх (де таргет порожній)
            historical_mask = working_df[target_col].notna() & (working_df[target_col] != "")
            
            train_df = working_df[historical_mask].copy()
            predict_df = working_df[~historical_mask].copy()
            
            if len(train_df) < 3:
                st.error("⚠️ Error: Not enough historical data rows with values to train the neural network. Please check your target column.")
            else:
                # Формуємо вхідні дані для навчання (X) та ціль (y)
                if use_feat and feature_col:
                    X_train = train_df[['Time_Index', feature_col]].values
                else:
                    X_train = train_df[['Time_Index']].values
                    
                y_train = train_df[target_col].values
                
                # Масштабування даних
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
                
                # Прогноз на всю таблицю
                if use_feat and feature_col:
                    X_all = working_df[['Time_Index', feature_col]].values
                else:
                    X_all = working_df[['Time_Index']].values
                    
                X_all_scaled = scaler_x.transform(X_all)
                y_all_pred_scaled = mlp.predict(X_all_scaled)
                working_df['Neural_Net_Forecast'] = scaler_y.inverse_transform(y_all_pred_scaled.reshape(-1, 1)).ravel()
                
                # Заповнюємо порожні клітинки майбутніх періодів прогнозованими значеннями
                working_df['Final_Output'] = working_df[target_col]
                missing_indices = working_df[target_col].isna() | (working_df[target_col] == "")
                working_df.loc[missing_indices, 'Final_Output'] = working_df.loc[missing_indices, 'Neural_Net_Forecast']

                st.success(t["success_msg"])
                st.dataframe(working_df)

                # Графік
                fig, ax = plt.subplots(figsize=(10, 4))
                # Історія
                ax.plot(train_df['Time_Index'], train_df[target_col], label="Historical Data (Actual)", color="black", marker="o")
                # Прогноз моделі
                ax.plot(working_df['Time_Index'], working_df['Neural_Net_Forecast'], label="Neural Network Model Forecast", color="purple", linestyle="--", marker="x")
                
                if use_feat and feature_col:
                    ax.plot(working_df['Time_Index'], working_df[feature_col], label="Plan / Feature", color="orange", linestyle=":")

                ax.set_title(t["chart_title"])
                ax.legend()
                ax.grid(True, linestyle=":", alpha=0.6)
                st.pyplot(fig)

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    working_df.to_excel(writer, index=False, sheet_name="NeuralNet_Forecast_Results")
                
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