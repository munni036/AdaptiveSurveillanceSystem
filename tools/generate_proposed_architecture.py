from PIL import Image, ImageDraw, ImageFont
import os

root = r"C:\Users\saran\OneDrive\Desktop\Da3_Btechh\literature_Review\AdaptiveSurveillanceResearch"
out_dir = os.path.join(root, "paper", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "proposed_architecture.png")

img = Image.new("RGB", (1600, 850), (245, 245, 245))
d = ImageDraw.Draw(img)

def box(x, y, w, h, fill, outline=(70, 70, 70), radius=12, width=2):
    d.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)

def text(s, x, y, size=26, fill=(38, 38, 38)):
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    d.text((x, y), s, font=font, fill=fill)

# Top-left surveillance stream
box(60, 220, 160, 120, (211, 211, 211))
text("Surveillance", 80, 250, 28)
text("Video Stream", 75, 285, 28)
text("(CCTV Camera)", 72, 320, 21)

# Preprocessing
box(245, 220, 170, 120, (211, 211, 211))
text("Preprocessing", 260, 250, 23)
text("Frame Extraction", 260, 285, 23)
text("Resize & Normalize", 258, 315, 19)
text("(Optional) Sampling", 262, 340, 19)

# Detection block
box(470, 140, 580, 255, (230, 230, 230), outline=(90, 90, 90), width=2)
text("Anomaly Detection (Inference)", 620, 125, 32)

# CAE blocks
box(500, 205, 190, 120, (211, 222, 236))
text("Trained Convolutional Autoencoder", 520, 220, 25)
box(540, 260, 110, 60, (220, 220, 220), outline=(70, 70, 70), width=2)
text("Encoder", 560, 282, 23)
text("E(.)", 576, 305, 22)
box(705, 205, 170, 120, (232, 214, 224))
text("Decoder", 745, 282, 23)
text("D(.)", 760, 305, 22)
text("Latent", 760, 245, 22)
text("Representation", 730, 260, 22)
text("z_t", 790, 315, 20)

# Reconstruction and threshold
box(905, 205, 170, 120, (240, 234, 216))
text("Reconstruction", 922, 240, 23)
text("Error", 970, 265, 23)
box(1085, 205, 170, 120, (240, 240, 220), outline=(80, 80, 80), width=2)
text("Threshold-based", 1105, 230, 23)
text("Decision", 1130, 255, 23)
text("Low", 1158, 285, 21)
text("Intermediate", 1122, 300, 21)
text("High", 1148, 325, 21)

# Candidate + human
box(1280, 205, 150, 120, (224, 224, 224), outline=(80, 80, 80), width=2)
text("Candidate", 1310, 230, 23)
text("Buffer", 1316, 255, 23)
text("Selected Frames", 1293, 285, 20)
text("Temporal", 1308, 305, 20)
text("Consistency", 1300, 325, 20)
box(1450, 205, 150, 120, (222, 214, 234))
text("Human", 1492, 230, 23)
text("Verification", 1475, 255, 23)
text("(Operator)", 1486, 285, 21)

# Right-side outcome boxes
box(1450, 95, 160, 70, (211, 211, 211))
text("Legitimate", 1475, 105, 24)
text("Scene Change", 1474, 130, 24)
box(1450, 350, 160, 70, (211, 211, 211))
text("Confirmed", 1486, 360, 24)
text("Anomaly", 1488, 385, 24)
box(1450, 470, 160, 70, (211, 211, 211))
text("Unresolved", 1482, 480, 24)

# Controlled adaptation box
box(500, 500, 720, 240, (245, 245, 245), outline=(90, 90, 90), width=2)
text("Controlled Model Adaptation", 760, 520, 32)

# Update candidate selection
box(520, 575, 170, 120, (211, 222, 236))
text("Update Candidate", 530, 594, 22)
text("Selection", 560, 618, 22)
text("Low-deviation", 534, 647, 20)
text("+ Verified Normal", 532, 666, 20)
text("+ Buffer Criteria", 532, 685, 20)

# Model update
box(760, 575, 170, 120, (240, 234, 216))
text("Model Update", 780, 594, 22)
text("Fine-tuning with", 770, 618, 22)
text("Selected Frames", 770, 642, 22)
text("+ Replay Loss", 778, 667, 22)

# Reference model comparison
box(1000, 575, 170, 120, (246, 229, 224))
text("Reference Model", 1014, 594, 22)
text("Comparison", 1028, 618, 22)
text("Drift Detection", 1014, 645, 20)
text("Performance Check", 1010, 668, 20)

# Accept or rollback
box(1240, 575, 170, 120, (220, 220, 220), outline=(80, 80, 80), width=2)
text("Accept Update", 1262, 594, 22)
text("or", 1304, 618, 22)
text("Rollback", 1280, 642, 22)
text("Based on Validation", 1260, 667, 22)
text("Criteria", 1288, 688, 22)

# Adaptive and reference models
box(1450, 575, 170, 120, (211, 211, 211))
text("Adaptive Model", 1470, 600, 22)
text("(M_t)", 1506, 624, 22)
text("(Updated)", 1492, 648, 22)
box(1450, 720, 170, 120, (211, 211, 211))
text("Reference Model", 1468, 742, 22)
text("(M_0)", 1502, 766, 22)
text("(Fixed)", 1490, 790, 22)

# Left accepted normal observations
box(60, 575, 170, 120, (211, 211, 211))
text("Accepted Normal", 72, 600, 22)
text("Observations", 80, 624, 22)

# Arrows
for x1, x2, y in [(220, 245, 280), (415, 470, 280), (1015, 1085, 265), (1230, 1280, 265), (1430, 1450, 265), (1430, 1450, 465), (1430, 1450, 610), (1430, 1450, 740)]:
    d.line((x1, y, x2, y), fill=(60, 60, 60), width=3)
    d.polygon([(x2, y), (x2 - 12, y - 7), (x2 - 12, y + 7)], fill=(60, 60, 60))

for x in [605, 740, 1065, 1275]:
    d.line((x, 500, x, 575), fill=(60, 60, 60), width=3)
    d.polygon([(x, 575), (x - 7, 575 - 12), (x + 7, 575 - 12)], fill=(60, 60, 60))
for x in [520, 760, 1000, 1240]:
    d.line((x, 695, x, 760), fill=(60, 60, 60), width=3)
    d.polygon([(x, 760), (x - 7, 760 - 12), (x + 7, 760 - 12)], fill=(60, 60, 60))
for x1, x2, y in [(230, 520, 635), (690, 760, 635), (930, 1000, 635), (1170, 1240, 635)]:
    d.line((x1, y, x2, y), fill=(60, 60, 60), width=3)
    d.polygon([(x2, y), (x2 - 12, y - 7), (x2 - 12, y + 7)], fill=(60, 60, 60))

# dashed borders around top and lower blocks
for x in range(470, 1100, 18):
    d.line((x, 120, x + 8, 120), fill=(90, 90, 90), width=2)
    d.line((x, 390, x + 8, 390), fill=(90, 90, 90), width=2)
for y in range(120, 390, 18):
    d.line((470, y, 470, y + 8), fill=(90, 90, 90), width=2)
    d.line((1100, y, 1100, y + 8), fill=(90, 90, 90), width=2)
for x in range(470, 1185, 18):
    d.line((x, 500, x + 8, 500), fill=(90, 90, 90), width=2)
    d.line((x, 740, x + 8, 740), fill=(90, 90, 90), width=2)
for y in range(500, 740, 18):
    d.line((470, y, 470, y + 8), fill=(90, 90, 90), width=2)
    d.line((1185, y, 1185, y + 8), fill=(90, 90, 90), width=2)

text("Discard", 1185, 220, 20)
text("from Update", 1160, 245, 20)
text("Accepted Normal", 85, 570, 20)
text("Observations", 90, 595, 20)

img.save(out_path)
print(f"Created {out_path}")
