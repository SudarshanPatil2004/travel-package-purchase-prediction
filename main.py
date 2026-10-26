import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Holiday Package Prediction", page_icon="✈️", layout="centered")

# Load models and preprocessor
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("preprocessor.joblib")
    models = joblib.load("model.joblib")
    # Extract the XGBoost model from the dictionary saved in the notebook
    model = models["XG Boost"] 
    return preprocessor, model

preprocessor, model = load_assets()

# Header
st.title("✈️ Holiday Package Purchase Prediction")
st.markdown("Enter the customer's details below to predict whether they will purchase the travel package.")

# UI/UX Layout - Grouped inputs into columns for better UX
with st.container():
    st.subheader("Customer Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        Gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        MaritalStatus = st.selectbox("Marital Status", ["Married", "Unmarried", "Divorced"])
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    with col3:
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=20000.0, step=500.0)

with st.container():
    st.subheader("Customer Engagement & Preferences")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        DurationOfPitch = st.number_input("Duration of Pitch (mins)", min_value=0.0, value=10.0, step=1.0)
        NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0.0, value=3.0, step=1.0)
    with col5:
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    with col6:
        PreferredPropertyStar = st.slider("Preferred Property Star", min_value=3.0, max_value=5.0, value=3.0, step=1.0)
        NumberOfTrips = st.number_input("Number of Previous Trips", min_value=0.0, value=2.0, step=1.0)
        TotalVisiting = st.number_input("Total Persons Visiting", min_value=1.0, value=3.0, step=1.0, help="Includes adults and children")

with st.container():
    st.subheader("Other Details")
    col7, col8 = st.columns(2)
    
    with col7:
        Passport_input = st.radio("Has Passport?", ["Yes", "No"])
        Passport = 1 if Passport_input == "Yes" else 0
    with col8:
        OwnCar_input = st.radio("Owns a Car?", ["Yes", "No"])
        OwnCar = 1 if OwnCar_input == "Yes" else 0

st.markdown("---")

# Predict Button
if st.button("Predict Package Purchase", use_container_width=True):
    # Construct DataFrame with Exact Column Names & Order expected by the preprocessor
    input_dict = {
        'Age': [Age],
        'TypeofContact': [TypeofContact],
        'CityTier': [CityTier],
        'DurationOfPitch': [DurationOfPitch],
        'Occupation': [Occupation],
        'Gender': [Gender],
        'NumberOfFollowups': [NumberOfFollowups],
        'ProductPitched': [ProductPitched],
        'PreferredPropertyStar': [PreferredPropertyStar],
        'MaritalStatus': [MaritalStatus],
        'NumberOfTrips': [NumberOfTrips],
        'Passport': [Passport],
        'PitchSatisfactionScore': [PitchSatisfactionScore],
        'OwnCar': [OwnCar],
        'Designation': [Designation],
        'MonthlyIncome': [MonthlyIncome],
        'TotalVisiting': [TotalVisiting]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    try:
        # Preprocess inputs
        transformed_data = preprocessor.transform(input_df)
        
        # Make Prediction
        prediction = model.predict(transformed_data)
        
        # Display Result
        st.markdown("### Prediction Result:")
        if prediction[0] == 1:
            st.success("🎯 High Probability: The customer is likely to purchase the package!")
            st.balloons()
        else:
            st.error("🛑 Low Probability: The customer is unlikely to purchase the package.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")