import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

def evaluate_metrics():
    # Define device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Define transform (must match training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load test data
    test_data = torchvision.datasets.ImageFolder(root="../resnet_images/test/", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
    class_names = test_data.classes
    num_classes = len(class_names)

    # Load model
    model = resnet50(weights=None)  # No pretrained weights
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    model_path = "saved_models/resnet50_finetuned.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Collect predictions, true labels, and probabilities
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metrics
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # Per-class precision & recall
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)

    # ROC AUC
    binarized_labels = label_binarize(all_labels, classes=list(range(num_classes)))
    try:
        roc_auc_macro = roc_auc_score(binarized_labels, all_probs, average='macro', multi_class='ovr')
        roc_auc_per_class = roc_auc_score(binarized_labels, all_probs, average=None, multi_class='ovr')
    except ValueError as e:
        roc_auc_macro = None
        roc_auc_per_class = None
        print(f"⚠️ ROC AUC computation failed: {e}")

    # Print results
    print(f"🧮 Balanced Accuracy: {balanced_acc:.4f}")
    print(f"🎯 Precision (macro): {precision_macro:.4f}")
    print(f"🔁 Recall (macro): {recall_macro:.4f}")

    print("\n📊 Per-Class Metrics:")
    for idx, cls in enumerate(class_names):
        print(f"Class: {cls}")
        print(f"  Precision: {precision_per_class[idx]:.4f}")
        print(f"  Recall:    {recall_per_class[idx]:.4f}")
        if roc_auc_per_class is not None:
            print(f"  ROC AUC:   {roc_auc_per_class[idx]:.4f}")
        print()

    if roc_auc_macro is not None:
        print(f"📈 ROC AUC (macro): {roc_auc_macro:.4f}")

    # Confusion matrix (raw)
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = confusion_matrix(all_labels, all_preds, normalize='true')

    # Plot raw confusion matrix
    disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp_raw.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix (Raw Counts)")
    plt.tight_layout()
    plt.show()

    # Plot normalized confusion matrix
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
    disp_norm.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_metrics()
