import os
import string
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from PIL import Image
import torchvision.transforms as T

API_KEY = os.environ.get("API_KEY", "change-me-now")
MODEL_PATH = "captcha_crnn_best.pth"
CHARS = string.digits + string.ascii_lowercase
BLANK = len(CHARS)
NUM_CLASSES = BLANK + 1
IMG_H, IMG_W = 60, 160
DEVICE = torch.device("cpu")

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        out, _ = self.rnn(conv)
        out = self.fc(out)
        return out.permute(1, 0, 2)

def greedy_decode(logits):
    idx = logits.argmax(2).cpu().numpy().T
    results = []
    for seq in idx:
        prev = -1
        s = []
        for i in seq:
            if i != prev and i != BLANK:
                s.append(CHARS[i])
            prev = i
        results.append("".join(s))
    return results

print("Loading model...")
model = CRNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded")

transform = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

app = FastAPI(title="CAPTCHA Solver API")

@app.get("/")
def root():
    return {"status": "ok", "message": "CAPTCHA Solver API"}

@app.post("/solve")
async def solve(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("L")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = greedy_decode(model(tensor))[0]

    return {"text": pred}\
