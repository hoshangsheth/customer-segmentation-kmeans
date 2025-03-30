import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        # Load your dataset (adjust the path to your file)
        file_path = '../data/marketing_campaign_final.csv'  # Update with the correct path
        data = pd.read_csv(file_path)
        
        # Select numeric columns for clustering
        features = data.select_dtypes(include=['float64', 'int64']).columns
        
        # Apply MinMaxScaler for feature scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[features])
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        
        return reduced_data, True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return a small dummy dataset if there's an error
        return np.random.rand(100, 2), False

# Streamlit app interface
st.title('Clustering Algorithm Comparison (Debug Version)')

# Load and preprocess data
reduced_data, data_loaded = load_data()

if data_loaded:
    st.success("Data loaded successfully!")
    st.write(f"Data shape: {reduced_data.shape}")
    
    # Debugging: Show a sample of the data
    st.subheader("Sample of PCA-reduced data:")
    st.dataframe(pd.DataFrame(reduced_data[:5], columns=["PC1", "PC2"]))
else:
    st.warning("Using dummy data for testing")

# Create sidebar for algorithm parameters
st.sidebar.title("Algorithm Parameters")

# KMeans parameters
st.sidebar.subheader("KMeans Parameters")
kmeans_n_clusters = st.sidebar.slider("KMeans Clusters", 2, 10, 3, key="sidebar_kmeans_n")

# DBSCAN parameters
st.sidebar.subheader("DBSCAN Parameters")
dbscan_eps = st.sidebar.slider("DBSCAN Epsilon", 0.1, 2.0, 0.5, step=0.05, key="sidebar_dbscan_eps")
dbscan_min_samples = st.sidebar.slider("DBSCAN Min Samples", 2, 20, 5, key="sidebar_dbscan_min")

# Agglomerative parameters
st.sidebar.subheader("Agglomerative Parameters")
agg_n_clusters = st.sidebar.slider("Agglomerative Clusters", 2, 10, 3, key="sidebar_agg_n")

# GMM parameters
st.sidebar.subheader("GMM Parameters")
gmm_n_components = st.sidebar.slider("GMM Components", 2, 10, 3, key="sidebar_gmm_n")

# Function to perform clustering
def run_clustering(algorithm, data, params):
    """Perform clustering with the given algorithm and parameters"""
    # Print parameters for debugging
    st.write(f"Running {algorithm} with parameters: {params}")
    
    if algorithm == 'KMeans':
        model = KMeans(n_clusters=params['n_clusters'], random_state=42)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    elif algorithm == 'Agglomerative Clustering':
        model = AgglomerativeClustering(n_clusters=params['n_clusters'])
    elif algorithm == 'Gaussian Mixture Model (GMM)':
        model = GaussianMixture(n_components=params['n_components'], random_state=42)
    
    # Fit the model
    labels = model.fit_predict(data)
    
    # Calculate silhouette score if there's more than one cluster
    unique_labels = set(labels)
    if len(unique_labels) > 1 and (not -1 in unique_labels or len(unique_labels) > 2):
        score = silhouette_score(data, labels)
    else:
        score = -1
        st.warning(f"{algorithm} produced only one effective cluster or mostly noise. Silhouette score not applicable.")
    
    return labels, score

# Run comparisons with explicit parameters
st.subheader("Algorithm Comparison with Debug Info")

# Collect all algorithm results in a list
results = []

# KMeans
kmeans_params = {'n_clusters': kmeans_n_clusters}
kmeans_labels, kmeans_score = run_clustering('KMeans', reduced_data, kmeans_params)
results.append({
    'Algorithm': 'KMeans',
    'Parameters': kmeans_params,
    'Score': kmeans_score,
    'Labels': kmeans_labels
})

# DBSCAN
dbscan_params = {'eps': dbscan_eps, 'min_samples': dbscan_min_samples}
dbscan_labels, dbscan_score = run_clustering('DBSCAN', reduced_data, dbscan_params)
results.append({
    'Algorithm': 'DBSCAN',
    'Parameters': dbscan_params,
    'Score': dbscan_score,
    'Labels': dbscan_labels
})

# Agglomerative
agg_params = {'n_clusters': agg_n_clusters}
agg_labels, agg_score = run_clustering('Agglomerative Clustering', reduced_data, agg_params)
results.append({
    'Algorithm': 'Agglomerative Clustering',
    'Parameters': agg_params,
    'Score': agg_score,
    'Labels': agg_labels
})

# GMM
gmm_params = {'n_components': gmm_n_components}
gmm_labels, gmm_score = run_clustering('Gaussian Mixture Model (GMM)', reduced_data, gmm_params)
results.append({
    'Algorithm': 'Gaussian Mixture Model (GMM)',
    'Parameters': gmm_params,
    'Score': gmm_score,
    'Labels': gmm_labels
})

# Create a DataFrame for the results
results_df = pd.DataFrame([
    {'Algorithm': r['Algorithm'], 'Parameters': str(r['Parameters']), 'Silhouette Score': r['Score']} 
    for r in results
])

# Show detailed results
st.subheader("Detailed Results:")
st.dataframe(results_df)

# Debugging: Count unique clusters for each algorithm
for result in results:
    unique_clusters = len(set(result['Labels']))
    non_noise_clusters = len(set(label for label in result['Labels'] if label != -1))
    st.write(f"{result['Algorithm']}: {unique_clusters} total clusters, {non_noise_clusters} non-noise clusters")

# Plot distribution of clusters for each algorithm
st.subheader("Cluster Distributions:")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, result in enumerate(results):
    # Count number of points in each cluster
    cluster_counts = pd.Series(result['Labels']).value_counts().sort_index()
    axes[i].bar(cluster_counts.index.astype(str), cluster_counts.values)
    axes[i].set_title(f"{result['Algorithm']} - Cluster Distribution")
    axes[i].set_xlabel("Cluster ID")
    axes[i].set_ylabel("Number of Points")
    
plt.tight_layout()
st.pyplot(fig)

# Visualization of clusterings
st.subheader("Clustering Visualizations:")
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes2 = axes2.flatten()

for i, result in enumerate(results):
    scatter = axes2[i].scatter(
        reduced_data[:, 0], 
        reduced_data[:, 1], 
        c=result['Labels'], 
        cmap='viridis', 
        marker='o'
    )
    axes2[i].set_title(f"{result['Algorithm']} (Score: {result['Score']:.4f})")
    axes2[i].set_xlabel('PC1')
    axes2[i].set_ylabel('PC2')
    fig2.colorbar(scatter, ax=axes2[i], label='Cluster')

plt.tight_layout()
st.pyplot(fig2)