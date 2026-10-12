import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.download_ped2 import download_ucsd_ped2
from src.data.ped2_dataset import UCSDPed2Dataset
from src.models.spatial_temporal_ae import SpatialTemporalAutoencoder
from src.buffer.safe_gate import SafeAdaptiveGate
from src.evaluate import calculate_and_plot_metrics

def main():
    print("===================================================================")
    print("   ADAPTIVE SURVEILLANCE SYSTEM USING DEEP LEARNING    ")
    print("===================================================================")

    # 1. Download / Verify UCSD Ped2 Dataset
    data_dir = download_ucsd_ped2()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Compute Hardware] Using Device: {device}")

    # 2. Prepare Data Loaders
    print("\n[Data Preparation] Loading UCSD Ped2 Dataset...")
    train_dataset = UCSDPed2Dataset(root_dir=data_dir, split="Train", image_size=(128, 128))
    test_dataset = UCSDPed2Dataset(root_dir=data_dir, split="Test", image_size=(128, 128))

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print(f"Dataset path notice: Preparing synthetic demo fallback if dataset folders are being populated.")
        from src.data.generate_demo_data import generate_demo_frames
        generate_demo_frames("data/demo_frames", num_frames=100)
        from src.data.dataset import VideoFrameDataset
        train_dataset = VideoFrameDataset("data/demo_frames", image_size=(128, 128))
        test_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"  Training Frames (Normal Baseline) : {len(train_dataset)}")
    print(f"  Testing Frames (With Anomalies)  : {len(test_dataset)}")

    # 3. Initialize Model & Optimizer
    model = SpatialTemporalAutoencoder(in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. Train Baseline Normality Model
    print("\n[Phase 1 Training] Learning Baseline Spatial-Temporal Normality...")
    epochs = 5
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for sample in train_loader:
            frames = sample[0] if isinstance(sample, (list, tuple)) else sample
            frames = frames.to(device)
            
            optimizer.zero_grad()
            reconstruction, _ = model(frames)
            loss = torch.nn.functional.mse_loss(reconstruction, frames)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * frames.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"  Epoch [{epoch+1}/{epochs}] | Reconstruction Loss: {epoch_loss:.6f}")

    # 5. Inference & Safe Adaptive Buffer Gating
    print("\n[Phase 1 Inference] Evaluating Safe Adaptive Gate & Uncertainty Banding...")
    model.eval()
    safe_gate = SafeAdaptiveGate(low_threshold=0.012, high_threshold=0.035)

    y_true = []
    y_scores = []
    
    pass_count, alert_count, buffer_count = 0, 0, 0

    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            frame = sample[0].to(device)
            label = sample[1].item() if len(sample) > 1 and isinstance(sample[1], torch.Tensor) else 0

            error, _ = model.compute_frame_error(frame)
            error_val = error.item()

            decision, score, msg = safe_gate.evaluate_frame_score(frame, error_val)

            y_true.append(label)
            y_scores.append(error_val)

            if decision == "PASS":
                pass_count += 1
            elif decision == "ALERT":
                alert_count += 1
            elif decision == "BUFFER":
                buffer_count += 1

            if idx % 50 == 0:
                print(f"  Frame [{idx:04d}] | Error Score: {error_val:.5f} | Decision: {decision}")

    # Simulate Sparse HITL Verification on Candidate Buffer
    print("\n[Sparse HITL Safety Check] Reviewing Candidate Buffer...")
    verified_samples, status_msg = safe_gate.process_human_verification(is_human_verified_benign=True)
    print(f"  {status_msg}")

    # 6. Evaluation & Metrics Output
    auc_score = calculate_and_plot_metrics(y_true, y_scores, output_dir="results")

    print("\n===================================================================")
    print("  SAFE ADAPTIVE SURVEILLANCE SYSTEM PIPELINE COMPLETED SUCCESSFULLY!  ")
    print("===================================================================")

if __name__ == "__main__":
    main()
