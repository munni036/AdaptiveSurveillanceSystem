import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.spatial_temporal_ae import SpatialTemporalAutoencoder
from src.buffer.safe_gate import SafeAdaptiveGate
from src.data.ped2_dataset import UCSDPed2Dataset
from src.evaluate import calculate_and_plot_metrics


def train_model(train_dataset, device):
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model = SpatialTemporalAutoencoder(in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n[Training] Learning baseline spatial-temporal normality...")
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for sample in train_loader:
            frames = sample[0] if isinstance(sample, (list, tuple)) else sample
            frames = frames.to(device)

            optimizer.zero_grad()
            reconstruction, _ = model(frames)
            loss = F.mse_loss(reconstruction, frames)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * frames.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"  Epoch [{epoch+1}/5] | Reconstruction Loss: {epoch_loss:.6f}")

    return model


def run_interactive_hitl_demo(buffer_capacity=50, review_count=5):
    print("==========================================================================")
    print("  HUMAN-IN-THE-LOOP (HITL) SAFE ADAPTATION DEMO                           ")
    print("==========================================================================")
    print("  Principle: 'Learn continuously, but not learn blindly.'                 ")
    print("==========================================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "data/UCSD_Ped2"
    if not os.path.exists(data_dir):
        print("UCSD Ped2 data not found. Please run main_pipeline.py first.")
        return

    train_dataset = UCSDPed2Dataset(root_dir=data_dir, split="Train", image_size=(128, 128))
    test_dataset = UCSDPed2Dataset(root_dir=data_dir, split="Test", image_size=(128, 128))

    print("[Stage 1] Training the model on normal frames...")
    model = train_model(train_dataset, device)
    model.eval()

    print("\n[Stage 2] Scoring test frames and populating buffer...")
    gate = SafeAdaptiveGate(low_threshold=0.015, high_threshold=0.040, max_buffer_capacity=buffer_capacity)
    buffered_events = []
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, (frame, label, img_path) in enumerate(test_loader):
            frame = frame.to(device)
            error, _ = model.compute_frame_error(frame)
            score = error.item()

            decision, score, msg = gate.evaluate_frame_score(frame, score)

            if decision == "BUFFER":
                buffered_events.append((idx, frame.squeeze(0), score, img_path[0]))

            if len(buffered_events) >= buffer_capacity:
                break

    if len(buffered_events) == 0:
        print("⚠️  No ambiguous frames were found. Try widening the uncertainty band or using a trained model with more variation.")
        return

    print(f"\n⚠️  Full candidate buffer collected: {len(buffered_events)} frames (max capacity = {buffer_capacity})")
    print(f"We will show only {min(review_count, len(buffered_events))} representative samples to the human operator.\n")

    review_samples = buffered_events[:review_count]
    accepted_frames = []
    rejected_count = 0
    accepted_count = 0

    print("[Stage 3] Human review on ambiguous buffered frames (Yes/No decision)")
    for item_idx, (frame_idx, frame_tensor, score, path) in enumerate(review_samples, start=1):
        filename = os.path.basename(path)
        print("--------------------------------------------------------------------------")
        print(f"Review Sample #{item_idx} | Frame [{frame_idx}] ({filename}) | Error: {score:.5f}")
        print("  Question: Is this a benign scene drift or a true anomaly?")

        user_input = input("  Human Operator: [y=yes / n=no] ").strip().lower()

        if user_input == 'y':
            accepted_count += 1
            accepted_frames.append(frame_tensor)
            print("  ✅ Verdict: BENIGN -> accepted for safe memory update.")
        else:
            rejected_count += 1
            print("  🚨 Verdict: ANOMALY -> discarded and memory protected.")

        print("--------------------------------------------------------------------------\n")

    print("[Stage 4] Safe memory update using only accepted benign samples")
    if len(accepted_frames) > 0:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for accepted_frame in accepted_frames:
            accepted_frame = accepted_frame.to(device).unsqueeze(0)
            optimizer.zero_grad()
            reconstruction, _ = model(accepted_frame)
            loss = F.mse_loss(reconstruction, accepted_frame)
            loss.backward()
            optimizer.step()
        print(f"  Safe update completed on {len(accepted_frames)} accepted benign frames.")
    else:
        print("  No benign samples were accepted; the model memory remains unchanged.")

    print("\n[Stage 5] Re-evaluating model performance after the human-guided update")
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for frame, label, _ in test_dataset:
            frame = frame.to(device).unsqueeze(0)
            error, _ = model.compute_frame_error(frame)
            y_true.append(label)
            y_scores.append(error.item())

    auc_score = calculate_and_plot_metrics(y_true, y_scores, output_dir="results")

    print("\n==========================================================================")
    print("  HITL REVIEW SUMMARY                                                    ")
    print("==========================================================================")
    print(f"  Candidate buffer size          : {buffer_capacity}")
    print(f"  Human-reviewed samples         : {min(review_count, len(buffered_events))}")
    print(f"  Accepted as benign             : {accepted_count}")
    print(f"  Rejected as anomalies          : {rejected_count}")
    print(f"  Final AUC-ROC after update     : {auc_score * 100:.2f}%")
    print("==========================================================================")
    print("  This demonstrates: uncertain frames are buffered, human vetting is used ")
    print("  only on ambiguous cases, and the model is updated safely from benign samples.")
    print("==========================================================================")


if __name__ == "__main__":
    run_interactive_hitl_demo(buffer_capacity=50, review_count=5)
