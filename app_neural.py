if st.button(t["train_btn"], type="primary"):
            working_df = df.copy()
            
            # Переконуємось, що числові колонки мають правильний формат
            working_df[target_col] = pd.to_numeric(working_df[target_col], errors='coerce')
            working_df[plan_col] = pd.to_numeric(working_df[plan_col], errors='coerce')
            
            working_df['Time_Index'] = range(len(working_df))
            
            # Історичні рядки: де є і plan, і demand (без NaN)
            train_df = working_df.dropna(subset=[target_col, plan_col]).copy()
            
            if len(train_df) < 3:
                st.error("⚠️ Error: Not enough valid historical data rows where both Plan and Demand are filled. Please check your data.")
            else:
                # Навчальна вибірка
                X_train = train_df[['Time_Index', plan_col]].values
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
                
                # Робимо прогноз для ВСІХ рядків таблиці (де є план)
                # Якщо в майбутніх рядках план заповнений, а demand порожній — мережа порахує туди прогноз
                valid_plan_mask = working_df[plan_col].notna()
                X_all = working_df.loc[valid_plan_mask, ['Time_Index', plan_col]].values
                
                X_all_scaled = scaler_x.transform(X_all)
                y_all_pred_scaled = mlp.predict(X_all_scaled)
                
                working_df['Neural_Net_Forecast'] = np.nan
                working_df.loc[valid_plan_mask, 'Neural_Net_Forecast'] = np.round(
                    scaler_y.inverse_transform(y_all_pred_scaled.reshape(-1, 1)).ravel(), 2
                )
                
                # Формуємо фінальну вихідну колоночку: для історії залишаємо факт, а для майбутнього вставляємо прогноз
                working_df['Final_Output'] = working_df[target_col]
                missing_indices = working_df[target_col].isna()
                working_df.loc[missing_indices, 'Final_Output'] = working_df.loc[missing_indices, 'Neural_Net_Forecast']

                st.success(t["success_msg"])
                st.dataframe(working_df)

                # Візуалізація результатів
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(working_df['Time_Index'], working_df[target_col], label="Historical Demand (Actual)", color="black", marker="o")
                ax.plot(working_df['Time_Index'], working_df['Neural_Net_Forecast'], label="Neural Network Forecast", color="purple", linestyle="--", marker="x")
                ax.plot(working_df['Time_Index'], working_df[plan_col], label="Plan", color="orange", linestyle=":")

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
