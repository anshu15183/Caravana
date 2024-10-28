import streamlit as st
import pandas as pd
import numpy as np
from streamlit_pages.streamlit_pages import MultiPage
import warnings
warnings.filterwarnings('ignore')
import os

st.set_page_config(
    page_title="Caravana",  # Set custom page  title
    page_icon="favicon.ico"
)

DATA_URL = pd.read_csv("Datasets/data.csv")  # Load your dataset

def home():
    st.title("CARAVANA")
    st.markdown("WELCOME TO CARAVANA ")
    
    padding = 0
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: 3rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)

    from PIL import Image
    st.subheader(" Why to use Caravana?")
    st.write(" \n Based on 3 C's : Collect , Calculate and Compare . \n It collects data from authentic source, calculates the maintenance cost of vehicle based on certain factors and then compares it with other vehicles. \n Compares data city-wise. \n Saves the time of the user browsing information and comparing them on various platforms. \n Shows the probability of specific components being changed.")
    st.subheader("How Caravana benefits everyone?")
    st.write(" \n Consumers : Enable to take a decision to buy a car based on the maintenance index. This enables the customer to select the car models according to the patterns and usage in zones, states, Cities, taluka. Further drilling down with respect to aging of vehicle, usage patterns and mileage \n Car Manufacturers : With the available data, OEM will be enable to design parts for more robustness. This will reduce warranty and service cost. Companies can analyze the performance of their models with respect to their competitors in a particular region.  \n Government : Enable govt for drafting the policies and approvals for new model launch in market. \n Second-hand Car Buyers : Customer will be able to take decision to buy the most suitable cars available for sale \n ")
    img=Image.open('Images/toyota.jpeg')
    st.image(img, width=445)
    image = Image.open('Images/mghector.jpeg')
    st.image(image, width=445)

def about():
    st.subheader("ABOUT CARAVANA")
    st.write("In today’s automotive landscape, with a plethora of cars and models available, determining which vehicle incurs higher maintenance costs can be quite challenging. While prospective buyers often invest considerable time evaluating various features of a car, the ongoing maintenance expenses are equally important yet often overlooked. Unfortunately, there is currently no dedicated platform that transparently presents the maintenance costs associated with different vehicles.")
    st.write("To address this gap, we have developed a system that provides comprehensive insights into the health and maintenance index of various car components and models. Our platform evaluates multiple factors to deliver an accurate assessment of the maintenance requirements for different car parts, including the likelihood of needing replacements after purchase.")
    st.write("Additionally, users can compare the maintenance costs of different car models, enabling potential buyers to make informed decisions based on expected maintenance expenses. This system not only assists consumers in understanding the long-term costs of owning a particular model but also provides valuable feedback to manufacturers regarding components that may require frequent servicing or replacement, encouraging improvements and better design in future models.")
    st.subheader('"Get to know your car\'s longevity prior to your purchase!"')
    st.text("\n")
    st.text("\n")
    st.text(" \n BY TEAM \n Anshu Singh \n Ashish Kumar \n Uttam Kumar \n Krishna Kumawat")

# New Calculator Page
def calculator():
    st.title("Car Maintenance Cost Calculator")
    st.write("Use this calculator to estimate the maintenance costs of your vehicle.")

    if st.button("Open Calculator"):
        # Run the second Streamlit app
        os.system("streamlit run home.py") 

app = MultiPage()
# Add pages
app.add_page("Home", home)
app.add_page("About", about)
app.add_page("Calculator", calculator)  # Added Calculator Page
app.run()
