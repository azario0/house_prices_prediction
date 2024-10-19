import customtkinter as ctk
import joblib
import pandas as pd

class PricePredictionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("House Price Prediction")
        self.geometry("400x600")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Load the saved XGBoost model, scaler, and feature names
        self.model = joblib.load('xgb_model.joblib')
        self.scaler = joblib.load('scaler.joblib')
        self.feature_names = joblib.load('feature_names.joblib')

        self.create_widgets()

    def create_widgets(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="House Price Prediction", font=("Arial", 20)).grid(row=0, column=0, pady=10)

        self.inputs = {}
        row = 1
        for feature in ['bedrooms', 'grade', 'has_basement', 'living_in_m2', 'renovated', 'nice_view', 'perfect_condition', 'real_bathrooms', 'has_lavatory', 'single_floor', 'year', 'month', 'quartile_zone']:
            ctk.CTkLabel(frame, text=feature.replace('_', ' ').title()).grid(row=row, column=0, pady=5, sticky="w")
            if feature in ['has_basement', 'renovated', 'nice_view', 'perfect_condition', 'has_lavatory', 'single_floor']:
                self.inputs[feature] = ctk.CTkCheckBox(frame, text="")
            elif feature in ['grade', 'month', 'quartile_zone']:
                self.inputs[feature] = ctk.CTkComboBox(frame, values=[str(i) for i in range(1, 5)] if feature == 'quartile_zone' else [str(i) for i in range(1, 14)])
            else:
                self.inputs[feature] = ctk.CTkEntry(frame)
            self.inputs[feature].grid(row=row, column=1, pady=5, sticky="ew")
            row += 1

        ctk.CTkButton(frame, text="Predict Price", command=self.predict_price).grid(row=row, column=0, columnspan=2, pady=20)

        self.result_label = ctk.CTkLabel(frame, text="")
        self.result_label.grid(row=row+1, column=0, columnspan=2, pady=10)

    def predict_price(self):
        input_data = {}
        for feature, widget in self.inputs.items():
            if isinstance(widget, ctk.CTkCheckBox):
                input_data[feature] = int(widget.get())
            elif isinstance(widget, ctk.CTkComboBox):
                input_data[feature] = int(widget.get())
            else:
                value = widget.get()
                input_data[feature] = float(value) if value else 0.0  # Handle empty values with a default
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_df, columns=['grade', 'month', 'quartile_zone'])

        # Ensure all features from the training set are present
        for feature in self.feature_names:
            if feature not in input_encoded.columns:
                input_encoded[feature] = 0

        # Reorder columns to match the training data
        input_encoded = input_encoded[self.feature_names]

        # Scale the input data
        input_scaled = self.scaler.transform(input_encoded)

        # Make prediction
        predicted_price = self.model.predict(input_scaled)[0]

        self.result_label.configure(text=f"Predicted Price: ${predicted_price:,.2f}")


if __name__ == "__main__":
    app = PricePredictionApp()
    app.mainloop()
