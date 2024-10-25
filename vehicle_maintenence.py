import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from streamlit_pages.streamlit_pages import MultiPage
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math
from socket import socket
from pandas import DataFrame


DATE_TIME = "date/time"
DATA_URL = pd.read_csv("Datasets/SIH.csv")
# DATA_URL = pd.read_csv("Datasets/generated_vehicle_data.csv")

def home():
    st.title("CARIFY")
    st.markdown("WELCOME TO CARIFY 💥")
    st.write("HOME")

    padding = 0
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)

    from PIL import Image
    st.write(" Why to use Carify?")
    st.text(" \n Based on 3 C's : Collect , Calculate and Compare . \n It collects data from authentic source, calculates the maintenance cost of vehicle based on certain factors and then compares it with other vehicles. \n Compares data city-wise. \n Saves the time of the user browsing information and comparing them on various platforms. \n Shows the probability of specific components being changed.")
    st.write("How Carify benefits everone?")
    st.text(" \n Consumers : Enable to take a decision to buy a car based on the maintenance index. This enables the customer to select the car models according to the patterns and usage in zones, states, Cities, taluka. Further drilling down with respect to aging of vehicle, usage patterns and mileage \n Car Manufacturers : With the available data, OEM will be enable to design parts for more robustness. This will reduce warranty and service cost. Companies can analyze the performance of their models with respect to their competitors in a particular region.  \n Government : Enable govt for drafting the policies and approvals for new model launch in market. \n Second-hand Car Buyers : Customer will be able to take decision to buy the most suitable cars available for sale \n ")
    img=Image.open('Images/toyota.jpeg')
    st.image(img, width=445)
    image = Image.open('Images/mghector.jpeg')
    st.image(image, width=445)
    ##st.write(DATA_URL,width=1000,height=1000)


# --------------------------------------- KNOW information about your CAR ---------------------------------------------------
def know():
    st.subheader("INFORMATION ABOUT YOUR CAR")
    select = st.selectbox('Company', ['Hyundai','Ford','Honda','KIA'])
    if select =='Hyundai':
        select1 = st.selectbox('Model', ['All New Santro', 'Creta', 'Grand i10','i20'])
    if select == 'Ford':
        select1 = st.selectbox('Model', ['Ecosport', 'Figo',])
    if select == 'Honda':
        select1 = st.selectbox('Model', ['Amaze', 'City(2014)', 'WR-V'])
    if select == 'KIA':
        select1 = st.selectbox('Model', ['Carnival'])

    select2 = st.selectbox('City', ['Mumbai','Delhi','Srinagar','Shimla','Vishakhapattnam'])
    select4 = st.selectbox('Fuel', ['Petrol','1.1 Petrol','1.2L Petrol','1.5L Petrol','Diesel','1.4L Diesel','1.5L Diesel','2.2L Diesel'])
    select3 = st.text_input('Enter Age of Vehicle here:')
    st.write("Check show data to get the information.")

    # ----------------------------------- PROBABILITY ------------------------------------------------
    #st.write(DATA_URL.columns)
    grouped=DATA_URL.groupby(['Company','Model','Fuel','City'])
    g=grouped.get_group((select,select1,select4,select2))
    st.write(g)

    from sklearn.model_selection import train_test_split
    d=g.loc[g["Age of Vehicle"] == select3]
    if(d['AC Dust Filter'].values == [0]):
           st.write("Probability of getting AC Dust Filter changed")
           st.write(0, "%")
    else:
        st.write("Probability of getting AC Dust Filter changed")
        st.write(100, "%")
    if(d['Engine oil'].values == [0]):
           st.write("Probability of getting Engine oil changed")
           st.write(0, "%")
    else:
        st.write("Probability of getting Engine oil changed")
        st.write(100, "%")
    if(d['Air cleaner filter'].values == [0]):
           st.write("Probability of getting Air cleaner filter changed")
           st.write(0, "%")
    else:
        st.write("Probability of getting Air cleaner filter changed")
        st.write(100, "%")
#-----------------------------------------------------------------------------------
    df=g

    X1 = df.iloc[:, 4:23].values
    y1 = df.iloc[:, 23].values
    df.head()
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.24, random_state=42)
    from sklearn.naive_bayes import GaussianNB as gnb

    ### create classifier
    clf = gnb()
    ### fit the classifier on the training features and labels
    clf.fit(X1_train, y1_train)
    y1_predict = clf.predict(X1_test)
    predictions3 = [np.round(value) for value in y1_predict]
    accuracy = accuracy_score(y1_test, predictions3)
    st.write("Drain washer Accuracy Probality: %.2f%%" % (accuracy * 100.0))

    #**************************************

    X1 = df.iloc[:, 4:22].values
    y1 = df.iloc[:, 22].values
    df.head()
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.24, random_state=42)
    from sklearn.naive_bayes import GaussianNB as gnb

    ### create classifier
    clf = gnb()
    ### fit the classifier on the training features and labels
    clf.fit(X1_train, y1_train)
    y1_predict = clf.predict(X1_test)
    predictions3 = [np.round(value) for value in y1_predict]
    accuracy = accuracy_score(y1_test, predictions3)
    st.write("Transmission fluid Probablity: %.2f%%" % (accuracy * 100.0))

    #***************************************

    X1 = df.iloc[:, 4:15].values
    y1 = df.iloc[:, 15].values
    df.head()
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.30, random_state=42)
    from sklearn.naive_bayes import GaussianNB as gnb

    ### create classifier
    clf = gnb()
    ### fit the classifier on the training features and labels
    clf.fit(X1_train, y1_train)
    y1_predict = clf.predict(X1_test)
    predictions3 = [np.round(value) for value in y1_predict]
    accuracy = accuracy_score(y1_test, predictions3)
    st.write("spark plug Probablity: %.2f%%" % (accuracy * 100.0))

    #****************************************

       
 # -------------------------------- COMPARISON ----------------------------------------------------------
def compare():

    import streamlit as st
    
    st.subheader("COMPARE TWO CARS")
    st.write("ENTER DETAILS OF CAR 1")
    selectt = st.selectbox('Company Name', ['Hyundai','Ford','Honda','KIA'])
    if selectt =='Hyundai':
        selectt1 = st.selectbox('Model Name', ['All New Santro', 'Creta', 'Grand i10','i20'])
    if selectt == 'Ford':
        selectt1 = st.selectbox('Model Name', ['Ecosport', 'Figo'])
    if selectt == 'Honda':
        selectt1 = st.selectbox('Model Name', ['Amaze', 'City (2014)', 'WR-V'])
    if selectt == 'KIA':
        selectt1 = st.selectbox('Model Name', ['Carnival'])

    selectt2 = st.selectbox('City', ['Mumbai','Delhi','Srinagar','Shimla','Vishakhapattnam'])
    selectt4 = st.selectbox('Fuel', ['Petrol','1.1 Petrol','1.2L Petrol','1.5L Petrol','Diesel','1.4L Diesel','1.5L Diesel','2.2L Diesel'])
    selectt3 = st.slider("Age of the vehicle.", 0, 200)

    st.write("ENTER DETAILS OF CAR 2")
    selecttt = st.selectbox('Company', ['Hyundai', 'Ford', 'Honda', 'KIA'])
    if selecttt == 'Hyundai':
        selecttt1 = st.selectbox('Model Name.', ['All New Santro', 'Creta', 'Grand i10', 'i20'])
    if selectt == 'Ford':
        selecttt1 = st.selectbox('Model Name.', ['Ecosport', 'Figo'])
    if selecttt == 'Honda':
        selecttt1 = st.selectbox('Model Name.', ['Amaze', 'City (2014)', 'WR-V'])
    if selectt == 'KIA':
        selecttt1 = st.selectbox('Model Name.', ['Carnival'])

    sel2 = st.selectbox('City', ['Mumbai', 'Delhi', 'Srinagar', 'Shimla', 'Vishakhapattnam'],key="10")
    sel4 = st.selectbox('Fuel', ['Petrol','1.1 Petrol','1.2L Petrol','1.5L Petrol','Diesel','1.4L Diesel','1.5L Diesel','2.2L Diesel'],key="11")
    sel3 = st.slider("Age of the vehicle", 0, 200)

    if st.button('Compare'):
        st.write("Comparing in progress!")
        grouped = DATA_URL.groupby(['Company', 'Model', 'Fuel', 'City'])
        g = grouped.get_group((selectt, selectt1, selectt4, selectt2))
        st.write(g)

        g1 = grouped.get_group((selecttt, selecttt1, sel4, sel2))
        st.write(g1)
 


# ------------------------------- VISUALISATION -----------------------------------------------------------
def visualize():
    fig, ax = plt.subplots()
    df=DATA_URL
    grouped1=df.groupby(['City'])
    height = grouped1['Total cost'].sum()
    st.write("COST vs CITY")
    plt.bar(np.arange(5), height)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(['Mumbai','Delhi','Srinagar','Shimla','Vishakhapattnam'])
    plt.xlabel('City')
    plt.ylabel('Total cost')
    st.plotly_chart(fig)


    fig, ax = plt.subplots()
    grouped1=df.groupby(['Company'])
    height = grouped1['Total cost'].sum()
    st.write("COST vs Company")  
    plt.bar(np.arange(4), height)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['Hyundai','Ford','Honda','KIA'])
    plt.xlabel('Company')
    plt.ylabel('Total cost')
    st.plotly_chart(fig)

 # ------------------------------- ABOUT US -----------------------------------------------------------
def about():
    st.subheader("ABOUT CARIFY")
    st.write("In Today’s world with so many cars, models in the market, it is hard to find out which car has a high maintenance cost/index that is authentic source. While buying a car we spend a lot of time different features and also maintenance. But there's no platform which tells us the maintenance cost that we'll have to pay after buying the car. A system that shows the health and Maintenance Index of various components of car models or car parts based on multiple factors is needed. We have created this platform for all these problems")
    st.write("In our system we show the health and maintenance Index of various components of car models or car parts based on multiple factors. Also the probablity of replacement of a particular car component after the car is bought. We can also compare two different car models based on maintenence. This will help new buyers to understand the maintenance costs of a certain model and probability of which car part requires more often servicing /change, OEM’s to understand which part is requiring frequent change so they can improve that component and needs to be recalled and made better in the new models ")
    st.subheader('"Know your car\'s life before you buy it!"')
    st.text("\n")
    st.text("\n")
    st.text(" \n BY TEAM \n Anshu Singh \n Ashish Kumar \n Uttam Kumar \n Krishna Kumawat")
    
app = MultiPage()
# Add pages
app.add_page("Home",home)
app.add_page("About",about)
app.add_page("Visualise",visualize)
app.add_page("Compare",compare)
app.add_page("Probablity",know)
app.run()
