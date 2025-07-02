from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("best.pt")  # Load your YOLOv8 model

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image)

    predictions = []
    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        predictions.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf,
            "class_id": int(cls),
            "label": results[0].names[int(cls)]
        })

    return JSONResponse(content={"results": predictions})
