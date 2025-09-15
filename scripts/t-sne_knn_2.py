import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
from collections import defaultdict
from PIL import Image
import joblib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



def load_npz_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    return embeddings, image_paths


def run_knn(knn_model_path, embeddings):
    knn = joblib.load(knn_model_path)
    return knn.predict(embeddings)


def parse_cluster_map(map_list):
    """
    Example input: ["0:5", "1:3", "2:8"]
    Output: {0:5, 1:3, 2:8}
    """
    mapping = {}
    for item in map_list:
        k, v = item.split(":")
        mapping[int(k)] = int(v)
    return mapping


def remap_predictions(predicted_labels, cluster_map):
    return [cluster_map.get(label, label) for label in predicted_labels]


def plot_tsne_with_images(
    embeddings, image_paths, labels,
    label_type='predicted', output_path=None,
    samples_per_label=1, perplexity=30, random_state=42,
    show_images=True
):
    print(f"🔍 Running T-SNE for {label_type} visualization...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(embeddings)

    print(f"🎨 Plotting T-SNE with {label_type} labels...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        tsne_results[:, 0], tsne_results[:, 1],
        c=labels, cmap='tab10', s=15, alpha=0.7
    )
    plt.colorbar(scatter, label=label_type)

    if show_images:
        print("🖼️ Adding image thumbnails...")
        label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            label_to_indices[label].append(i)

        for label_id, indices in label_to_indices.items():
            chosen = np.random.choice(indices, size=min(samples_per_label, len(indices)), replace=False)
            for idx in chosen:
                try:
                    img = Image.open(image_paths[idx]).convert('RGB')
                    img.thumbnail((40, 40), Image.Resampling.LANCZOS)
                    img_np = np.asarray(img)

                    imagebox = OffsetImage(img_np, zoom=1)
                    ab = AnnotationBbox(
                        imagebox,
                        (tsne_results[idx, 0], tsne_results[idx, 1]),
                        frameon=False
                    )
                    plt.gca().add_artist(ab)
                except Exception as e:
                    print(f"⚠️ Could not load image {image_paths[idx]}: {e}")

    else:
        print("📊 Showing scatter only (no image overlays)")

    plt.title(f"T-SNE with {label_type.capitalize()} Labels (from KNN)")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize predicted clusters using a trained KNN model")
    parser.add_argument('--embedding_file', type=str, required=True, help="Path to .npz file with embeddings and image paths")
    parser.add_argument('--knn_model', type=str, required=True, help="Path to saved KNN model (.joblib)")
    parser.add_argument('--output_file', type=str, default="tsne_knn_predictions.png")
    parser.add_argument('--samples_per_label', type=int, default=1)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--show_images', action='store_true', help="If set, overlays sample images on the plot")
    parser.add_argument('--cluster_to_label', nargs='+', help="Optional mapping from cluster IDs to class labels (e.g., 0:5 1:3)")

    args = parser.parse_args()

    embeddings, image_paths = load_npz_data(args.embedding_file)
    predicted_labels = run_knn(args.knn_model, embeddings)

    label_type = 'KNN'

    if args.cluster_to_label:
        cluster_map = parse_cluster_map(args.cluster_to_label)
        predicted_labels = remap_predictions(predicted_labels, cluster_map)
        label_type = 'remapped'

    plot_tsne_with_images(
        embeddings=embeddings,
        image_paths=image_paths,
        labels=predicted_labels,
        label_type=label_type,
        output_path=args.output_file,
        samples_per_label=args.samples_per_label,
        perplexity=args.perplexity,
        show_images=args.show_images
    )


if __name__ == "__main__":
    main()
