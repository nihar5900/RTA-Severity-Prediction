import streamlit as st
import numpy as np
from process import ordinal,manualOrdinal,get_prediction
import joblib

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")

model = joblib.load(r'model/tune_model_to_impliment.pkl')

#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']

options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']

options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']

options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']

type_of_collision = ['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision',
                     'Collision with roadside objects', 'Collision with animals', 'Other', 'Rollover', 
                     'Fall from vehicles', 'Collision with pedestrians', 'With Train', 'Unknown']

option_sevice_yr = ['Above 10yr', '5-10yrs', '1-2yr', '2-5yrs', 'Unknown', 'Below 1yr']

# Accedent sevirity
aSeverity={'Slight Injury':0,'Serious Injury':1,'Fatal injury':2}

# For Driver Age
aDriver={'Unknown':0,'Under 18':1,'18-30':2,'31-50':3,'Over 51':4}

#For Driver Experience
dExp={'unknown':0,'No Licence':1,'Below 1yr':2,'1-2yr':3,'2-5yr':4,'5-10yr':5,'Above 10yr':6}

#For Vechile sevice year
vService={'Unknown':0,'Below 1yr':1,'1-2yr':2,'2-5yrs':3,'5-10yrs':4,'Above 10yr':5}

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('predictin_form'):
        st.subheader("Enter the below Features:")

        hour=st.slider("Pickup Hours: ",0,23,value=0,format="%d")
        days_of_week=st.selectbox("Select Days of The Week: ",options=options_day)
        driver_age=st.selectbox("Driver's Age: ",options=options_age)
        driver_exp=st.selectbox("Driver Experience: ",options=options_driver_exp)
        vechile_involve=st.slider("Number of Vechiles Involve: ",1,7,format="%d")
        casulities=st.slider("Number of casualities: ",1,8,format="%d")
        vechile_type=st.selectbox("Vechile Type: ",options=options_vehicle_type)
        vechile_servis_yr=st.selectbox("Servise Year of the Vechile: ",options=option_sevice_yr)
        area_acc=st.selectbox("Area of Accedent: ",options=options_acc_area)
        lines=st.selectbox("Lanes: ",options=options_lanes)
        type_collision=st.selectbox("Collision Type: ",options=type_of_collision)

        submit=st.form_submit_button("Predict")
    if submit:
        hour=hour
        days_of_week=ordinal(days_of_week,options_day)
        driver_age=manualOrdinal(driver_age,aDriver)
        driver_exp=manualOrdinal(driver_exp,dExp)
        vechile_involve=vechile_involve
        casulities=casulities
        vechile_type=ordinal(vechile_type,options_vehicle_type)
        vechile_servis_yr=manualOrdinal(vechile_servis_yr,vService)
        area_acc=ordinal(area_acc,options_acc_area)
        lines=ordinal(lines,options_lanes)
        type_collision=ordinal(type_collision,type_of_collision)

        data = np.array([hour,days_of_week,driver_age,driver_exp,vechile_involve,
                         casulities,driver_age,vechile_type,vechile_servis_yr,
                         area_acc,lines,type_collision]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        outpt = list(filter(lambda x: aSeverity[x] == pred, aSeverity))[0]

        st.write(f"The predicted severity is:  {outpt}")

if __name__=='__main__':
    main()