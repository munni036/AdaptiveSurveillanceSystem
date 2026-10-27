import torch

class CandidateBuffer:
    """
    Uncertainty-driven Candidate Buffer and Safe Write-Gate.
    Prevents memory poisoning by isolating ambiguous/novel patterns until 
    temporal stability or sparse human verification authorizes adaptation.
    """
    def __init__(self, low_threshold=0.02, high_threshold=0.08, max_buffer_size=100):
        self.low_threshold = low_threshold    # Below this: High Confidence Normal
        self.high_threshold = high_threshold  # Above this: High Confidence Anomaly
        self.max_buffer_size = max_buffer_size
        self.buffer = []

    def evaluate_frame(self, frame_tensor, error_score):
        """
        Classifies frame error score into decision zones:
        - PASS: High Confidence Normal (Auto-update)
        - ALERT: High Confidence Anomaly (High priority alert)
        - BUFFER: Ambiguous / Novel pattern (Hold for verification)
        """
        if error_score < self.low_threshold:
            return "PASS", "High Confidence Normal - Auto Update Allowed"
        elif error_score > self.high_threshold:
            return "ALERT", "High Confidence Anomaly - High Priority Alert"
        else:
            # Store in candidate buffer
            if len(self.buffer) >= self.max_buffer_size:
                self.buffer.pop(0)  # FIFO queue
            self.buffer.append((frame_tensor, error_score))
            return "BUFFER", f"Ambiguous Pattern - Stored in Buffer (Queue Size: {len(self.buffer)})"

    def get_buffered_candidates(self):
        return self.buffer

    def flush_buffer(self):
        self.buffer.clear()
