import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster")

def extract_damage_coordinates(predicted_mask, damage_threshold=3):
    if hasattr(predicted_mask, 'cpu'):
        mask_array = predicted_mask.cpu().numpy()
    else:
        mask_array = predicted_mask
    y_coords, x_coords = np.where(mask_array >= damage_threshold)
    return np.column_stack((y_coords, x_coords))

def cluster_rescue_zones(coordinates, num_zones=3):
    if len(coordinates) < num_zones:
        return None, None
    kmeans = KMeans(n_clusters=num_zones, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coordinates)
    return kmeans.cluster_centers_, cluster_labels

def apply_pca_reduction(tabular_features_df):
    pca = PCA(n_components=2) 
    return pca.fit_transform(tabular_features_df)