import torch

class SafeAdaptiveGate:
    """
    Uncertainty-Driven Candidate Buffer & Safe Write-Gate.
    
    Operates under the principle: 'Learn continuously, but not learn blindly.'
    - PASS: High-Confidence Normal (Low Error) -> Safe to update memory.
    - ALERT: High-Confidence Anomaly (High Error) -> High-priority alarm triggered.
    - BUFFER: Ambiguous / Novel Scene Shift -> Placed in candidate buffer. 
      Only verified benign frames update memory, preventing anomaly contamination.
    """
    def __init__(self, low_threshold=0.015, high_threshold=0.045, max_buffer_capacity=50):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.max_buffer_capacity = max_buffer_capacity
        self.candidate_buffer = []

    def evaluate_frame_score(self, frame_tensor, reconstruction_error):
        """
        Evaluates reconstruction error score against uncertainty decision thresholds.
        """
        score = float(reconstruction_error)

        if score <= self.low_threshold:
            return "PASS", score, "High Confidence Normal - Safe Auto-Update"
        elif score >= self.high_threshold:
            return "ALERT", score, "High Confidence Anomaly - Priority Alarm Triggered"
        else:
            # Ambiguous pattern zone -> Isolate in candidate buffer
            if len(self.candidate_buffer) >= self.max_buffer_capacity:
                self.candidate_buffer.pop(0)  # FIFO queue
            self.candidate_buffer.append((frame_tensor.cpu(), score))
            return "BUFFER", score, f"Ambiguous Drift - Held in Candidate Buffer (Size: {len(self.candidate_buffer)})"

    def process_human_verification(self, is_human_verified_benign=True):
        """
        Sparse Human-in-the-Loop Verification Safety Check.
        If operator confirms benign scene shift -> Release frames for safe memory update.
        If operator flags anomaly -> Flush buffer to prevent memory poisoning.
        """
        verified_samples = []
        if is_human_verified_benign:
            verified_samples = [item[0] for item in self.candidate_buffer]
            status_msg = f"Human Verified: {len(verified_samples)} buffered frames committed to memory."
        else:
            status_msg = f"Human Flagged Anomaly: {len(self.candidate_buffer)} buffered frames flushed. Memory protected!"

        self.candidate_buffer.clear()
        return verified_samples, status_msg
