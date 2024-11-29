import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

#loading in model
try:
    model = joblib.load('house_price_model.pkl')
    scaler = joblib.load('scaler.pkl')  # Load the scaler

except FileNotFoundError as e:
    raise FileNotFoundError(f"Required file not found: {e}")


#function for the prediction button
def on_predict_button_click():
    try:
        # Gather inputs from the UI
        size = float(size_entry.get())  # Entry for property size
        bedrooms = int(bedrooms_entry.get())
        bathrooms = int(bathrooms_entry.get())
        living_rooms = int(living_rooms_entry.get())

        # Create the input array for the model
        inputs = np.array([size, bedrooms, bathrooms, living_rooms])

        # Debugging: Print input vector shape to ensure correctness
        print(f"Input vector shape: {inputs.shape}")

        # Predict using the model
        prediction_log = model.predict(inputs)[0]
        prediction = np.expm1(prediction_log)  # Reverse log transformation

        # Display the prediction in the GUI
        result_label.config(text=f"Predicted Price: £{prediction:,.2f}")

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", str(e))

#preprocessing the features
def preprocess_inputs(size, bedrooms, bathrooms, living_rooms):
    #initialize a dictionary with default values for all features
    input_dict = {'bathrooms' : bathrooms, 'bedrooms' : bedrooms, 'floorAreaSqM' : size, 'livingRooms' : living_rooms}

    #creating a DataFrame with features in the correct order
    aligned_inputs_df = pd.DataFrame([input_dict])

    #scaling the inputs
    return scaler.transform(aligned_inputs_df)

def predict_price():
    try:
        #get the user inputs
        size = float(size_entry.get())
        bedrooms = int(bedrooms_entry.get())
        bathrooms = int(bathrooms_entry.get())
        living_rooms = int(living_rooms_entry.get())

        #preprocess inputs
        inputs = preprocess_inputs(size, bedrooms, bathrooms, living_rooms)

        #debugging
        print(f"Input vector shape: {inputs.shape}")  

        #making prediction
        prediction_log = model.predict(inputs)[0]  # Get the prediction in log scale
        prediction = np.expm1(prediction_log)  # Reverse the log transformation

        #display the prediction in the GUI
        result_label.config(text=f"Predicted Price: £{prediction:,.2f}")

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", str(e))



# Create the GUI
root = tk.Tk()
root.title("House Price Predictor")

# Size
size_label = tk.Label(root, text="Size (sq ft):")
size_label.grid(row=0, column=0, padx=10, pady=5)
size_entry = tk.Entry(root)
size_entry.grid(row=0, column=1, padx=10, pady=5)

# Bedrooms
bedrooms_label = tk.Label(root, text="Number of Bedrooms:")
bedrooms_label.grid(row=1, column=0, padx=10, pady=5)
bedrooms_entry = tk.Entry(root)
bedrooms_entry.grid(row=1, column=1, padx=10, pady=5)

# Bathrooms
bathrooms_label = tk.Label(root, text="Number of Bathrooms:")
bathrooms_label.grid(row=2, column=0, padx=10, pady=5)
bathrooms_entry = tk.Entry(root)
bathrooms_entry.grid(row=2, column=1, padx=10, pady=5)

# Living Rooms
living_rooms_label = tk.Label(root, text="Number of Living Rooms:")
living_rooms_label.grid(row=3, column=0, padx=10, pady=5)
living_rooms_entry = tk.Entry(root)
living_rooms_entry.grid(row=3, column=1, padx=10, pady=5)


# Predict Button
predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.grid(row=5, column=0, columnspan=2, pady=10)

# Result
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.grid(row=6, column=0, columnspan=2, pady=10)

# Get the coefficients (importance) of each feature
coefficients = model.coef_

#feature importance
importance = np.abs(coefficients)
sorted_idx = np.argsort(importance)

#run the application
root.mainloop()
