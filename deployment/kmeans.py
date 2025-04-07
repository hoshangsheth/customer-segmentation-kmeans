import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

# Set Streamlit page configuration
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# Load model and preprocessing tools
with open("deployment/kmeans.pkl", "rb") as model_file:
    kmeans = pickle.load(model_file)
with open("deployment/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("deployment/pca.pkl", "rb") as pca_file:
    pca = pickle.load(pca_file)
with open("deployment/cluster_mapping.pkl", "rb") as mapping_file:
    cluster_mapping = pickle.load(mapping_file)

# Custom CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1E1E2F; padding-top: 20px; }
        .menu-container { background: linear-gradient(135deg, #FF6B6B, #FF8E53); padding: 15px; border-radius: 10px; }
        .stButton>button { border-radius: 8px; background-color: #ff6b6b; color: white; padding: 10px 24px; }
        .stButton>button:hover { background-color: #ff4757; }
        h1, h2, h3 { color: #ff6b6b; }
    </style>
""", unsafe_allow_html=True)

image_url = "https://cdn-icons-png.flaticon.com/512/1239/1239719.png"

# Sidebar Menu
with st.sidebar:
    st.markdown("""
    <style>
        .header-container {
            background: black; /* Black Background */
            border-radius: 15px;
            padding: 10px 15px;
            text-align: left;
            color: white; /* White Text */
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 10px;
            border: 2px solid black; /* Black Border (Hidden) */
            width: fit-content;
        }
        .header-container img {
            width: 30px;
            height: 30px;
        }
    </style>
                """, unsafe_allow_html=True)
    
    st.markdown(f"""
                <div class="header-container">
                <img src="{image_url}" />
                <span>Customer Segmentation</span>
                </div>
                """, unsafe_allow_html=True)


    selected = option_menu("Main Menu", ["Home", "Customer Segmentation", "Insights & Trends", "Meet the Team", "Contact Us"], 
                           icons=["house", "search", "bar-chart", "people", "envelope"], menu_icon="menu-app", default_index=0)
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Initialize session state variables if not already set
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = None
if "predicted_segment" not in st.session_state:
    st.session_state.predicted_segment = None

# Page Navigation Handling
if selected == "Home":
    st.title("Optimized Customer Segmentation for Business Growth")
    st.write("Leverage Machine Learning to analyze customer segment, enhance targeting strategies, and maximize profitability.")

    container = st.container(border=True)
    container.subheader("Customer Segments 👥")
    container.write("Luxury Shoppers:")
    container.write("These are your most valuable customers who make frequent and large purchases. Target them with loyalty programs, exclusive discounts, and premium offers to maximize retention and revenue.")
    container.write("Budget-Conscious Buyers:")
    container.write("These customers engage less frequently or spend minimally. Use personalized promotions, strategic upselling, and engagement campaigns to increase their lifetime value.")
    
    with st.popover('Instructionsℹ️'):
        st.markdown("How the model works:")
        st.subheader("1️⃣ Customer Segmentation:")
        st.write("Users input customer features on 'Customer Segmentation Page' such as Income, Number of Kids (Kidhome), Number of Teens (Teenhome), " \
        "Monthly Spending on Wines, Meat, Fish, Number of Web Visits, Age, Total Campaigns Engaged, and Purchase Frequency.")
        st.subheader("2️⃣ AI-Powered Prediction:")
        st.write("The KMeans Clustering model analyzes inputs and classifies customers into High-Spending or Low-Spending segments.")
        st.subheader("3️⃣ Insights & Trends:")
        st.write("View bar and pie charts to visualize spending trends and customer distribution for data-driven decision-making.")



    
elif selected == "Customer Segmentation":
    st.title("Customer Segmentation Prediction")
    st.write("Enter customer details to predict their segment.")
    
    income = st.number_input("Customer's annual income", min_value=1000, step=1)
    kidhome = st.selectbox("Kids at Home", list(range(6)))
    teenhome = st.selectbox("Teens at Home", list(range(6)))
    mnt_wines = st.slider("Amount spent on Wine", min_value=0, max_value=1000, step=10)
    mnt_meat = st.slider("Amount spent on Meat", min_value=0, max_value=1000, step=10)
    mnt_fish = st.slider("Amount spent on Fish", min_value=0, max_value=1000, step=10)
    web_visits = st.slider("Web Visits per Month", min_value=0, max_value=100, step=1)
    age = st.slider("Age", min_value=18, max_value=100, step=1)
    total_campaigns = st.slider("Total Campaigns Participated", min_value=0, max_value=4, step=1)
    purchase_freq = st.number_input("Purchase Frequency", min_value=0, step=1)
    
    
    if st.button("Predict Segment"):
        # Define recommendations
        recommendations = {"Luxury Shopper": "Offer VIP memberships, personalized shopping experiences, and exclusive discounts.",
                           "Budget-Conscious Buyer": "Run flash sales, bundle discounts, and promote budget-friendly product bundles."}
        user_data = np.array([[income, kidhome, teenhome, mnt_wines, mnt_meat, mnt_fish, web_visits, age, total_campaigns, purchase_freq]])
        user_scaled = scaler.transform(user_data)
        user_pca = pca.transform(user_scaled)
        predicted_cluster = kmeans.predict(user_pca)[0]
        segment = cluster_mapping[predicted_cluster]
        st.success(f"Predicted Segment: {segment}")
        st.info(f'Marketing Recommendations: {recommendations[segment]}')

        # ✅ Store results in session state
        st.session_state.user_inputs = {
            'income': income, 'mnt_wines': mnt_wines, 'mnt_meat': mnt_meat,
            'mnt_fish': mnt_fish, 'web_visits': web_visits, 'age': age
            }
        st.session_state.predicted_segment = segment

elif selected == "Insights & Trends":
    st.title("Latest Predicted Customer Spending Insights")
    
    if st.session_state.predicted_segment is None:
        st.write("Please predict a segment first in the Customer Segmentation page.")
    else:
        st.subheader(f"Predicted Segment: {st.session_state.predicted_segment}")
        st.write(f"Customer's Income: {st.session_state.user_inputs['income']}")
        
        input_data = pd.DataFrame({
            'Category': ['Wine Spending', 'Meat Spending', 'Fish Spending', 'Web Visits'],
            'Values': [
                st.session_state.user_inputs['mnt_wines'],
                st.session_state.user_inputs['mnt_meat'],
                st.session_state.user_inputs['mnt_fish'],
                st.session_state.user_inputs['web_visits']
            ]
        })
        
        st.subheader("Spending Overview")
        st.bar_chart(input_data.set_index('Category'))

        st.subheader("Spending Distribution")
        pie_data = pd.DataFrame({
            'Category': ['Wine', 'Meat', 'Fish'],
            'Spending': [
                st.session_state.user_inputs['mnt_wines'],
                st.session_state.user_inputs['mnt_meat'],
                st.session_state.user_inputs['mnt_fish']
                ]
                })
        fig = px.pie(pie_data, names='Category', values='Spending', 
                     title="Proportion of Spending on Different Products",
                     hole=0.3)  # Adds a slight donut effect for better visualization
        st.plotly_chart(fig)

elif selected == "Meet the Team":
    team_container = st.container(border=True)
    team_container.title("The Team 👨‍💻👩‍💻")
    team_members = {
        "Hoshang": "[GitHub](https://github.com/Hoshhh08)",
        "Paranidhran": "[GitHub](https://github.com/baranidharan27)",
        "Koushik Guduru": "[GitHub](https://github.com/koushikGuduru)",
        "Sarayu": "[GitHub](https://github.com/Sarayu1903)",
        "Gayatri": "[GitHub](https://github.com/gayatri-robo)"
    }
    for name, github in team_members.items():
        st.write(f"👤 {name}: {github}")
    
    describe_container = st.container(border=True)
    describe_container.write("Our team collaborates seamlessly to refine data-driven strategies, " \
    "ensuring accurate customer segmentation and marketing recommendations. " \
    "Through collective expertise and innovation, we optimize every step—from model " \
    "training to deployment—for maximum efficiency and effectiveness.")

elif selected == "Contact Us":
    st.title("Contact Us 📬")
    team_emails = {"Hoshang": "hoshang@example.com", "Paranidhran": "paranidhran@example.com", 
                   "Koushik Guduru": "koushik@example.com", "Sarayu": "sarayu@example.com", 
                   "Gayatri": "gayatri@example.com"}
    selected_member = st.selectbox("Select Team Member", list(team_emails.keys()))
    email = team_emails[selected_member]
    name = st.text_input("Your Name")
    user_email = st.text_input("Your Email")
    message = st.text_area("Your Message")
    if st.button("📨 Send Message"):
        if name and user_email and message:
            st.success(f"✅ Message sent to {selected_member} at {email}!")
        else:
            st.error("⚠️ Please fill out all fields before sending.")
