import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Apply global average pooling to convert (N, D, H, W) to (N, D)")
    parser.add_argument('--input_file', type=str, required=True, help="Path to original .npz file with 4D embeddings")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the reshaped 2D embeddings")
    args = parser.parse_args()

    data = np.load(args.input_file, allow_pickle=True)
    embeddings = data['embeddings']  # shape: (N, 2048, 7, 7)
    image_paths = data['image_paths']

    print("Original shape:", embeddings.shape)

    if embeddings.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {embeddings.shape}")

    # Global average pooling over spatial dims (H, W)
    pooled = embeddings.mean(axis=(2, 3))  # shape: (N, 2048)

    print("Pooled shape:", pooled.shape)

    np.savez(args.output_file, embeddings=pooled, image_paths=image_paths)
    print(f"✅ Saved reshaped embeddings to {args.output_file}")

if __name__ == "__main__":
    main()
