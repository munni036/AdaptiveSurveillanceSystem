import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def calculate_and_plot_metrics(y_true, y_scores, output_dir="results"):
    """
    Computes Frame-Level Area Under ROC Curve (AUC-ROC) 
    and plots evaluation curves saved to results/.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"\n==========================================================")
    print(f"  SAFE ADAPTIVE SURVEILLANCE SYSTEM - EVALUATION METRICS  ")
    print(f"==========================================================")
    print(f"  Frame-Level AUC-ROC Score : {roc_auc * 100:.2f}%")
    print(f"  Total Evaluated Frames   : {len(y_true)}")
    print(f"  Normal Frames Evaluated  : {sum(1 for y in y_true if y == 0)}")
    print(f"  Anomalous Frames Flagged : {sum(1 for y in y_true if y == 1)}")
    print(f"==========================================================")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Safe Adaptive VAD (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
    plt.title('Frame-Level ROC Curve - UCSD Ped2 Benchmark', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    roc_plot_path = os.path.join(output_dir, "auc_roc_curve.png")
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC Curve plot saved to: '{roc_plot_path}'")
    return roc_auc
