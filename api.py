import os
from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from torchvision import transforms


# -----------------------------
# Model settings
# -----------------------------

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
BLANK = len(CHARS)
NUM_CLASSES = BLANK + 1

IMG_H = 60
IMG_W = 160

DEVICE = torch.device("cpu")


# -----------------------------
# CRNN model
# -----------------------------

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(512, 512, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.rnn = nn.LSTM(
            512,
            256,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)

        # [B, C, 1, W] -> [B, W, C]
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)
        x = self.fc(x)

        return x.permute(1, 0, 2)


# -----------------------------
# Load model
# -----------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "captcha_crnn_best.pth"
)

model = CRNN().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()


# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# -----------------------------
# Decode prediction
# -----------------------------

def decode(logits):
    predictions = logits.argmax(2)

    result = []
    previous = BLANK

    for token in predictions[:, 0]:
        token = token.item()

        if token != previous and token != BLANK:
            result.append(CHARS[token])

        previous = token

    return "".join(result)


# -----------------------------
# FastAPI
# -----------------------------

app = FastAPI()

API_KEY = os.environ.get("MODEL_API_KEY")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/predict")
async def predict(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="MODEL_API_KEY is not configured"
        )

    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    image_data = await file.read()

    try:
        image = Image.open(BytesIO(image_data)).convert("L")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)

    prediction = decode(logits)

    return {
        "prediction": prediction
    }
