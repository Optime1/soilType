from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from soil_model import SoilCNN, CLASSES  # Убедитесь, что soil_model.py доступен
import io

app = FastAPI(title="Soil Classification ML Service")

# Загрузка модели
model = SoilCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load('soil_model_4classes.pth', map_location='cpu'))
model.eval()

# Трансформация для inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict_soil(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
    
    soil_type = CLASSES[top_catid[0].item()]
    confidence = top_prob[0].item() * 100
    return {"soil_type": soil_type, "confidence": f"{confidence:.2f}%"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)