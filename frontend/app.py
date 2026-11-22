
# Import required libraries for UI development, data handling, and API communication
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Configure the application title and display introductory instructions
st.title("SuperKart System")
st.write(
    "Enter the product and store details below to predict the total sales."
)

# Collect numerical and categorical product and store features through Streamlit input widgets
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store", "Food Mart"])
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

# Assemble the collected inputs into a dictionary matching the model's expected feature names
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Handle single-record sales prediction when the Predict button is clicked
if st.button("Predict", type='primary'):

    # Send product and store details as JSON to the Flask single-prediction endpoint
    response = requests.post(
        f"{BACKEND_URL}/v1/predict",
        json=product_data
    )

    # Display the predicted sales if the API request is successful
    if response.status_code == 200:
        result = response.json()
        predicted_sales = result["Sales"]
        st.success(f"Predicted Product Store Sales Total: ₹{predicted_sales:.2f}")
    else:
        # Display an error message if the API request fails
        st.error("Unable to connect to the prediction API.")

# Display the section for batch sales prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file containing multiple product-store records
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

# Process batch predictions when a file is uploaded and the batch button is clicked
if uploaded_file is not None:

    if st.button("Predict for Batch", type='primary'):

        # Send the uploaded CSV file to the Flask batch-prediction endpoint
        response = requests.post(
            f"{BACKEND_URL}/v1/predictbatch",
            files={"file": uploaded_file}
        )

        # Process and display the batch prediction results if the request succeeds
        if response.status_code == 200:
            results = response.json()

            # Display a success message after completing batch predictions
            st.success("Predictions completed successfully!")

            try:
                # Convert list-based API responses into a Pandas DataFrame
                if isinstance(results, list):
                    df = pd.DataFrame(results)
                elif isinstance(results, dict):
                    # Check if all values are scalars
                    if all(not isinstance(v, (list, dict)) for v in results.values()):
                        df = pd.DataFrame([results])
                    else:
                        df = pd.DataFrame(results)
                else:
                    # Handle other response formats by converting them into a DataFrame
                    df = pd.DataFrame({"Result": [results]})

                # Display batch prediction results in an interactive table
                st.dataframe(df, use_container_width=True)

            # Display raw JSON output if DataFrame conversion fails
            except Exception as e:
                st.error(f"Unable to display results as a table: {e}")
                st.json(results)

        else:
            # Display an error message if the batch prediction API request fails
            st.error("Unable to connect to the prediction API.")
