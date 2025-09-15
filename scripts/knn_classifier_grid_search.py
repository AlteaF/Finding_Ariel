import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_mutual_info_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
import joblib
import os
import re
import csv
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import pandas as pd
import seaborn as sns


def load_npz_data(npz_path):
    return np.load(npz_path, allow_pickle=True)


def extract_ground_truth_labels(image_paths):
    true_labels = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]
        tokens = re.findall(r'\d+', basename)
        if not tokens:
            print(f"⚠️ Warning: No numeric token found in filename: {filename}")
            true_labels.append(-1)
            continue
        class_id = int(tokens[-1])
        true_labels.append(class_id)
    return true_labels


def compute_ami(predicted_labels, image_paths):
    true_labels = extract_ground_truth_labels(image_paths)
    paired = [(c, t) for c, t in zip(predicted_labels, true_labels) if t != -1]
    if not paired:
        raise ValueError("No valid labels found for AMI.")
    c_labels, t_labels = zip(*paired)
    return adjusted_mutual_info_score(t_labels, c_labels)


def show_sample_predictions(image_paths, true_labels, predicted_labels, num_samples=10):
    print(f"\n🖼️ Showing {num_samples} sample predictions...")
    indices = np.random.choice(len(image_paths), size=min(num_samples, len(image_paths)), replace=False)
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        try:
            img = Image.open(image_paths[idx]).convert("RGB")
        except Exception as e:
            print(f"⚠️ Could not open image {image_paths[idx]}: {e}")
            continue
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pred: {predicted_labels[idx]}\nTrue: {true_labels[idx]}")
    plt.tight_layout()
    plt.show()


def parse_cluster_map(map_list):
    return {int(k): int(v) for item in map_list for k, v in [item.split(":")]}


def remap_predictions(predicted_labels, cluster_map):
    return [cluster_map.get(label, label) for label in predicted_labels]


def compute_per_class_accuracy(predicted_labels, image_paths, output_csv=None):
    true_labels = extract_ground_truth_labels(image_paths)

    class_totals = defaultdict(int)
    class_correct = defaultdict(int)

    for pred, true in zip(predicted_labels, true_labels):
        if true == -1:
            continue
        class_totals[true] += 1
        if pred == true:
            class_correct[true] += 1

    per_class_stats = []
    for cls in sorted(class_totals):
        total = class_totals[cls]
        correct = class_correct[cls]
        acc = 100.0 * correct / total if total > 0 else 0.0
        per_class_stats.append({
            'Class': cls,
            'Correct': correct,
            'Total': total,
            'Accuracy (%)': round(acc, 2)
        })

    df = pd.DataFrame(per_class_stats)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"📊 Per-class accuracy saved to {output_csv}")
    else:
        print(df.to_string(index=False))


def evaluate_predictions(y_true, y_pred, class_names=None, output_prefix=None):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"\n🎯 F1 Scores:\n  Micro: {f1_micro:.4f}\n  Macro: {f1_macro:.4f}\n  Weighted: {f1_weighted:.4f}")

    if output_prefix:
        class_names = class_names or [str(i) for i in sorted(set(y_true))]
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(f"{output_prefix}_confusion_matrix.csv")
        cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
        cm_norm_df.to_csv(f"{output_prefix}_confusion_matrix_normalized.csv")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm_df, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Normalized Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_confusion_matrix_normalized.png")
        plt.close()

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{output_prefix}_classification_report.csv")

        f1_scores_df = pd.DataFrame({
            "Metric": ["Micro F1", "Macro F1", "Weighted F1"],
            "Score": [f1_micro, f1_macro, f1_weighted]
        })
        f1_scores_df.to_csv(f"{output_prefix}_f1_scores.csv", index=False)


def extract_label_names(mapping):
    reverse_map = {v: k for k, v in mapping.items()}
    sorted_true_classes = sorted(mapping.values())
    return [f"Class {label}" for label in sorted_true_classes]


def main(args):
    print("📥 Loading training data...")
    train_data = load_npz_data(args.train_embeddings)
    X_train = train_data['embeddings']
    image_paths_train = train_data['image_paths']
    y_train = train_data['cluster_labels']

    print("📥 Loading validation/test embedding file...")
    full_data = load_npz_data(args.val_test_embeddings)
    X_all = full_data['embeddings']
    image_paths_all = full_data['image_paths']
    y_all = np.array(extract_ground_truth_labels(image_paths_all))

    valid_indices = [i for i, label in enumerate(y_all) if label != -1]
    X_all = X_all[valid_indices]
    image_paths_all = image_paths_all[valid_indices]
    y_all = y_all[valid_indices]

    X_val, X_test, y_val, y_test, paths_val, paths_test = train_test_split(
        X_all, y_all, image_paths_all,
        test_size=args.val_test_split_ratio,
        stratify=y_all,
        random_state=42
    )

    ami_scores = []
    best_ami = -1
    best_k = None
    best_model = None
    best_predictions = None

    print("🔍 Testing K values:", args.k_list)
    for k in args.k_list:
        print(f"\n🚀 Testing k={k}")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        val_preds = knn.predict(X_val)

        if args.cluster_to_label:
            cluster_map = parse_cluster_map(args.cluster_to_label)
            val_preds = remap_predictions(val_preds, cluster_map)
        else:
            cluster_map = {}

        ami = adjusted_mutual_info_score(y_val, val_preds)
        print(f"✅ AMI for k={k}: {ami:.4f}")
        ami_scores.append({'k': k, 'AMI': ami})

        if ami > best_ami:
            best_ami = ami
            best_k = k
            best_model = knn
            best_predictions = val_preds

    print(f"\n🏆 Best K: {best_k} with AMI: {best_ami:.4f}")

    if args.save_model:
        joblib.dump(best_model, args.save_model)
        print(f"💾 Best model saved to {args.save_model}")

    if args.ami_csv:
        with open(args.ami_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['k', 'AMI'])
            writer.writeheader()
            writer.writerows(ami_scores)

    if args.per_class_csv:
        compute_per_class_accuracy(best_predictions, paths_val, args.per_class_csv)

    show_sample_predictions(paths_val, y_val, best_predictions)

    if args.eval_prefix:
        evaluate_predictions(y_val, best_predictions, class_names=extract_label_names(cluster_map), output_prefix=args.eval_prefix)

    # 🔍 Evaluate on test set
    print("\n📥 Evaluating on test data with best model...")
    test_preds = best_model.predict(X_test)
    if args.cluster_to_label:
        test_preds = remap_predictions(test_preds, cluster_map)

    test_ami = adjusted_mutual_info_score(y_test, test_preds)
    print(f"✅ Test AMI with best k={best_k}: {test_ami:.4f}")

    show_sample_predictions(paths_test, y_test, test_preds)

    if args.per_class_csv:
        test_csv = args.per_class_csv.replace(".csv", "_test.csv")
        compute_per_class_accuracy(test_preds, paths_test, test_csv)

    if args.eval_prefix:
        test_prefix = args.eval_prefix + "_test"
        evaluate_predictions(y_test, test_preds, class_names=extract_label_names(cluster_map), output_prefix=test_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KNN classification with AMI-based model selection and test evaluation")
    parser.add_argument('--train_embeddings', type=str, required=True, help='Path to .npz with train embeddings, image paths, and cluster labels')
    parser.add_argument('--val_test_embeddings', type=str, required=True, help='Path to .npz file to split into validation and test sets')
    parser.add_argument('--val_test_split_ratio', type=float, default=0.5, help='Fraction to allocate to test (default 0.5)')
    parser.add_argument('--k_list', nargs='+', type=int, required=True, help='List of K values to test (e.g. --k_list 1 3 5)')
    parser.add_argument('--save_model', type=str, required=True, help='Path to save best KNN model')
    parser.add_argument('--ami_csv', type=str, help='Path to save AMI results as CSV')
    parser.add_argument('--cluster_to_label', nargs='+', help='Mapping from cluster index to true label, e.g. 0:2 1:11 2:1')
    parser.add_argument('--per_class_csv', type=str, help='Path to save per-class accuracy CSV')
    parser.add_argument('--eval_prefix', type=str, help='Prefix path for confusion matrix and classification report')

    args = parser.parse_args()
    main(args)
