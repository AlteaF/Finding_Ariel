import torch
import timm
from transformers import AutoModel
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import argparse

def extract_embeddings(image_dir, model, device, transform):
    """Extract embeddings for all images in the given directory."""
    image_paths = []
    embeddings = []

    # Loop through all image files in the directory
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        # Check if it is a valid image file (you can add more checks for different formats)
        if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
            print(image_path)
            image_paths.append(image_path)
            image = Image.open(image_path).convert("RGB")

            # Apply the necessary transformations
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Forward pass to extract features (embeddings)
            with torch.no_grad():
                outputs = model(image_tensor)
                feature = outputs[0]

            # Append the embeddings
            embeddings.append(feature.cpu().numpy())

    return np.array(embeddings), image_paths

def save_embeddings(embeddings, image_paths, output_file):
    """Save embeddings and image paths to a file."""
    np.savez(output_file, embeddings=embeddings, image_paths=image_paths)
    print(f"Embeddings saved to {output_file}")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract MegaDescriptor embeddings from images in a directory.")
    parser.add_argument("--image_dir", type=str, help="Directory containing images to process.")
    parser.add_argument("--output_file", type=str, help="File path to save the extracted embeddings.")
    args = parser.parse_args()

    # Load MegaDescriptor model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #model = timm.create_model("hf_hub:BVRA/MegaDescriptor-T-224", pretrained=True) #tiny, 28M params
    model = timm.create_model("hf_hub:BVRA/MegaDescriptor-B-224", pretrained=True) #base, 109M params
    #model = timm.create_model("hf_hub:BVRA/MegaDescriptor-L-224", pretrained=True) #large, 228M params
    model = model.eval().to(device)

    # Define the necessary transformations manually (resizing, normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the input size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for MegaDescriptor
    ])

    # Extract embeddings
    embeddings, image_paths = extract_embeddings(args.image_dir, model, device, transform)

    #print(embeddings.shape)
    #print(embeddings)

    # Save the embeddings
    save_embeddings(embeddings, image_paths, args.output_file)

if __name__ == "__main__":
    main()

