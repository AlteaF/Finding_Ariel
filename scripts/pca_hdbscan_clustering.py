import numpy as np
import argparse
import hdbscan
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm

def run_pca(embeddings, n_components=50):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca

def run_hdbscan(embeddings, min_cluster_size=10):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embeddings)
    return labels, clusterer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', required=True, help='Path to .npz file with embeddings and image_paths')
    parser.add_argument('--output_file', required=True, help='Output .npz file with cluster labels and paths')
    parser.add_argument('--min_cluster_size', type=int, default=10)
    parser.add_argument('--pca_components', type=int, default=50)
    args = parser.parse_args()

    print("Loading embeddings...")
    data = np.load(args.embedding_file, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']

    print(f"Running PCA to {args.pca_components} dimensions...")
    reduced, pca_model = run_pca(embeddings, args.pca_components)
    joblib.dump(pca_model, "pca_model.joblib")

    print("Clustering with HDBSCAN...")
    labels, clusterer = run_hdbscan(reduced, args.min_cluster_size)

    np.savez(args.output_file, labels=labels, image_paths=image_paths, cluster_labels=labels)
    print(f"Saved clustered data to {args.output_file}")

if __name__ == '__main__':
    main()
