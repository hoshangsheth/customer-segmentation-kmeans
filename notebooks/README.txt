# Project: Customer Segmentation Model

## Overview
This project uses machine learning to segment customers into two categories:
1. **Luxury Shoppers**
2. **Budget-Conscious Buyers**

It provides personalized marketing recommendations based on customer spending patterns.

## Features
- **Customer Segmentation** using KMeans clustering
- **Dimensionality Reduction** with PCA
- **Interactive Dashboard** built with Streamlit
- **Data Visualization** using Plotly
- **User-friendly Interface** with an Option Menu

## Requirements
Ensure you have the required libraries installed. Install dependencies using:
```sh
pip install -r requirements.txt
```

## How to Run
1. Clone the repository:
```sh
git clone <your-repo-url>
```
2. Navigate to the project directory:
```sh
cd your-project-folder
```
3. Run the application:
```sh
streamlit run app.py
```

## Deployment
To deploy on **Streamlit Cloud**:
1. Push your project to GitHub
2. Connect GitHub to Streamlit Cloud
3. Deploy with `requirements.txt` included

## How It Works
1. Users input customer data through the app interface.
2. The model processes the data and classifies customers into **Luxury Shoppers** or **Budget-Conscious Buyers**.
3. Personalized marketing insights and recommendations are displayed on the dashboard.
4. Users can gain insights from the visualizations to explore trends and patterns.

