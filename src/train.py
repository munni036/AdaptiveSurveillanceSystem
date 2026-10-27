import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import VideoFrameDataset
from src.models.autoencoder import SpatialAutoencoder
from src.buffer.candidate_buffer import CandidateBuffer

def run_training_and_evaluation(data_dir="data/demo_frames", epochs=5, lr=0.001):
    print("==========================================================")
    print("  Safe Adaptive Surveillance System - Execution Pipeline  ")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # 1. Load Dataset
    print(f"\n1. Loading frames from '{data_dir}'...")
    dataset = VideoFrameDataset(folder_path=data_dir, image_size=(128, 128), is_grayscale=True)
    if len(dataset) == 0:
        print(f"Error: No images found in '{data_dir}'. Please check dataset path!")
        return

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"Total Frames Loaded: {len(dataset)}")

    # 2. Initialize Model & Optimizer
    print("\n2. Initializing SpatialAutoencoder Model...")
    model = SpatialAutoencoder(in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Model Training
    print("\n3. Training Baseline Normality Model...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for frames, label, img_path in dataloader:
            frames = frames.to(device)
            optimizer.zero_grad()
            reconstruction, _ = model(frames)
            loss = torch.nn.functional.mse_loss(reconstruction, frames)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * frames.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"  Epoch [{epoch+1}/{epochs}] - Normality Reconstruction Loss: {epoch_loss:.6f}")

    # 4. Inference & Candidate Buffer Evaluation
    print("\n4. Running Inference & Candidate Buffer Evaluation...")
    model.eval()
    buffer = CandidateBuffer(low_threshold=0.015, high_threshold=0.040)

    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    pass_count, alert_count, buffer_count = 0, 0, 0

    with torch.no_grad():
        for idx, (frame, label, img_path) in enumerate(test_dataloader):
            frame = frame.to(device)
            error, _ = model.compute_reconstruction_error(frame)
            error_val = error.item()

            decision, msg = buffer.evaluate_frame(frame, error_val)

            if decision == "PASS":
                pass_count += 1
            elif decision == "ALERT":
                alert_count += 1
            elif decision == "BUFFER":
                buffer_count += 1

            if idx % 20 == 0 or decision == "ALERT":
                frame_name = os.path.basename(img_path[0])
                print(f"  Frame [{idx:03d} - {frame_name}] | Error: {error_val:.5f} | Action: {decision}")

    print("\n==========================================================")
    print("                  EVALUATION SUMMARY                      ")
    print("==========================================================")
    print(f"  Total Processed Frames  : {len(dataset)}")
    print(f"  Auto-Passed (Normal)   : {pass_count}")
    print(f"  Buffered (Ambiguous)   : {buffer_count}")
    print(f"  High Priority Alerts   : {alert_count}")
    print("==========================================================")

if __name__ == "__main__":
    run_training_and_evaluation()
