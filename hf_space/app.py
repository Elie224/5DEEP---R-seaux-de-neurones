from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Workaround for Gradio 4.44.0 schema bug on Spaces
try:
    import gradio_client.utils as _gc_utils

    _orig_get_type = _gc_utils.get_type
    _orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type

    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "any"
        return _orig_get_type(schema)

    def _patched_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_schema_to_python_type(schema, defs)

    _gc_utils.get_type = _patched_get_type
    _gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
except Exception:
    pass


class CNNB15M(nn.Module):
    """Architecture inferred from saved state_dict (Q4 best model variant B_1p5M)."""

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_package.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model package not found: {MODEL_PATH}")

package = torch.load(MODEL_PATH, map_location="cpu")
labels = package.get("labels", ["buildings", "forest", "glacier", "mountain", "sea", "street"])
image_size = tuple(package.get("taille_image", [160, 160]))
acc_test = package.get("accuracy_test", None)

model = CNNB15M(num_classes=len(labels))
model.load_state_dict(package["state_dict"], strict=True)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
)


def predict(image: Image.Image):
    if image is None:
        return "Aucune image chargee."

    x = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = int(torch.argmax(probs).item())
    ranked = sorted(
        [(labels[i], float(probs[i].item())) for i in range(len(labels))],
        key=lambda x: x[1],
        reverse=True,
    )
    lines = [
        f"Prediction: {labels[top_idx]} | Confiance: {probs[top_idx].item() * 100:.2f}%",
        "",
        "Top-6 probabilites:",
    ]
    lines.extend([f"- {name}: {score * 100:.2f}%" for name, score in ranked])
    return "\n".join(lines)


description = (
    "Modele CNN from-scratch exporte depuis le notebook du projet. "
    "Charge une image et observe les probabilites par classe."
)
if acc_test is not None:
    description += f"\n\nAccuracy test du modele exporte: {acc_test * 100:.2f}%"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Image d'entree"),
    outputs=gr.Textbox(label="Resultat", lines=10),
    title="Classification de scenes (Intel Image)",
    description=description,
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
