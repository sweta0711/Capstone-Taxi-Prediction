import streamlit as st
import pickle
import numpy as np

lin_model=pickle.load(open('Linear_model.pkl','rb'))
#random_model=pickle.load(open('random_model.pkl','rb'))
model_xgb=pickle.load(open('XGB_model.pkl','rb'))




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
    
    
    a=st.text_input('ft_5')
    b=st.text_input('ft_4')
    c=st.text_input('ft_3')
    d=st.text_input('ft_2')
    e=st.text_input('ft_1')
    f=st.text_input('freq1')
    g=st.text_input('freq2')
    h=st.text_input('freq3')
    i=st.text_input('freq4')
    j=st.text_input('freq5')
    k=st.text_input('Amp1')
    l=st.text_input('Amp2')
    m=st.text_input('Amp3')
    n=st.text_input('Amp4')
    o=st.text_input('Amp5')
    p=st.text_input('Latitude')
    q=st.text_input('Longitude')
    r=st.text_input('WeekDay')
    s=st.text_input('WeightedAvg')
    

   
    inputs=np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s]])
    
    
    if st.button('Predict'):#button name is Classify
        if option == 'Linear Regression':
            inputs = inputs.astype(np.float)
            st.success(lin_model.predict(inputs))
        elif option == 'Xgboost':
            inputs = inputs.astype(np.float)
            st.success(model_xgb.predict(inputs))
        #else:
          #  st.success(random_model.predict(inputs))
            
                
            

if __name__=='__main__':
    main()
        
    
            
    
    