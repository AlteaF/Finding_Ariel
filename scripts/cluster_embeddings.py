import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
import argparse
import matplotlib.image as mpimg
from PIL import Image
import os

def load_embeddings(embedding_file):
    """Load the saved embeddings and image paths from the file"""
    data = np.load(embedding_file, allow_pickle=True)
    embeddings = data['embeddings']
    if(len(embeddings.shape) == 3):
        embeddings = np.squeeze(embeddings, axis=1)
    image_paths = data['image_paths']
    #print(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings, image_paths

def perform_clustering(embeddings, n_clusters=5):
    """Perform KMeans clustering on the embeddings"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans

def get_rescaled_image(image_path, target_size=(100, 100)):
    """Rescale the image to a fixed size"""
    img = Image.open(image_path)
    #img_rescaled = img.resize(target_size, Image.ANTIALIAS) #older version does not work anymore
    img_rescaled = img.resize(target_size, Image.Resampling.LANCZOS)

    return img_rescaled

def create_mosaic_for_clusters(image_paths, cluster_labels, output_dir="mosaics", target_size=(100, 100)):
    """Create and save mosaic of images for each cluster"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_clusters = len(set(cluster_labels))

    # Create mosaic for each cluster
    for cluster_id in range(n_clusters):
        # Get all images in the current cluster
        cluster_image_paths = [image_paths[i] for i in range(len(image_paths)) if cluster_labels[i] == cluster_id]
        
        # Rescale images and add them to the mosaic
        mosaic_images = []
        for image_path in cluster_image_paths:
            img_rescaled = get_rescaled_image(image_path, target_size=target_size)
            mosaic_images.append(img_rescaled)

        # Determine grid size for the mosaic
        n_images = len(mosaic_images)
        cols = int(np.ceil(np.sqrt(n_images)))  # number of columns
        rows = int(np.ceil(n_images / cols))  # number of rows

        # Create an empty mosaic array
        mosaic_array = np.zeros((rows * target_size[1], cols * target_size[0], 3), dtype=np.uint8)

        # Add the images to the mosaic array
        for i, img in enumerate(mosaic_images):
            row = i // cols
            col = i % cols
            mosaic_array[row * target_size[1]:(row + 1) * target_size[1], col * target_size[0]:(col + 1) * target_size[0]] = np.array(img)

        # Convert the mosaic to an image and save it
        mosaic_img = Image.fromarray(mosaic_array)
        mosaic_img.save(os.path.join(output_dir, f"cluster_{cluster_id}_mosaic.jpg"))
        print(f"Mosaic for cluster {cluster_id} saved!")


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Read saved embeddings, perform clustering, create and save image mosaics for each cluster.")
    parser.add_argument('--embedding_file', type=str, help="Path to the saved embedding file.")
    parser.add_argument('--n_clusters', type=int, default=5, help="Number of clusters for KMeans.")
    parser.add_argument('--output_dir', type=str, default="mosaics", help="Directory to save mosaics.")
    args = parser.parse_args()
    
    # Load embeddings and image paths
    embeddings, image_paths = load_embeddings(args.embedding_file)
    
    # Perform clustering
    cluster_labels, kmeans = perform_clustering(embeddings, n_clusters=args.n_clusters)
    
    # Create and save mosaics
    create_mosaic_for_clusters(image_paths, cluster_labels, output_dir=args.output_dir, target_size=(100, 100)) #, rows=5, cols=5)

if __name__ == "__main__":
    main()
