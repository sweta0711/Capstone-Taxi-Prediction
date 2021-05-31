import streamlit as st
import pickle
import numpy as np

lin_model=pickle.load(open('Linear_model.pkl','rb'))
random_model=pickle.load(open('random_model.pkl','rb'))
model_xgb=pickle.load(open('model_xgb.pkl','rb'))




def main():
    st.title("Taxi Demand Prediction")
    html_temp = """
    <div style="background-color:teal ; padding:10px">
    <h2 style="color:white; text-align:center;">Taxi Demand Prediction </h2>
    </div>
    """
    
    
    
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Linear Regression','Xgboost','Random Forest']
    option=st.sidebar.selectbox('Select the model',activities)
    st.subheader(option)
    
    
    a=st.text_input('VendorID')
    b=st.text_input('passenger_count')
    c=st.text_input('trip_distance')
    d=st.text_input('pickup_longitude')
    e=st.text_input('pickup_latitude')
    f=st.text_input('RateCodeID')
    g=st.text_input('store_and_fwd_flag')
    h=st.text_input('dropoff_longitude')
    i=st.text_input('dropoff_latitude')
    j=st.text_input('payment_type')
    k=st.text_input('fare_amount')
    l=st.text_input('extra')
    m=st.text_input('mta_tax')
    n=st.text_input('tip_amount')
    o=st.text_input('tolls_amount')
    p=st.text_input('improvement_surcharge')
    

   
    inputs=np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]])
    inputs = inputs.astype(np.float)
    
    if st.button('Predict'):#button name is Classify
        if option == 'Linear Regression':
            st.success(lin_model.predict(inputs))
        elif option == 'Xgboost':
            st.success(model_xgb.predict(inputs))
        else:
            st.success(random_model.predict(inputs))
            
                
            

if __name__=='__main__':
    main()
        
    
            
    
    