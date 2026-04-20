from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Используем новую CPU-модель
model_path = "best_resnet_last_two_layers_cpu.pkl"
model = None

try:
    state_dict = torch.load(model_path, map_location='cpu')
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(state_dict)
    model.eval()
    print("Модель загружена")
except Exception as e:
    print(f"Ошибка: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.get('/status')
def status():
    model_status = "загружена" if model is not None else "ошибка загрузки"
    return {"status": "Сервис работает", "model": model_status}

@app.get('/version')
def version():
    return {"model": "ResNet18 (last two layers)", "classes": 2}


@app.post('/predict')
async def predict(image_id: str = Query(...), file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Только изображения")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            result = predicted.item()
            if int(result) == 1:
                result_text = "Реальное изображение"
            elif int(result) == 0:
                result_text = "Сгенерированное нейросетями"
            else:
                result_text = "Ошибка"

        class_names = ["Real", "AI_generated"]

        return {
            "image_id": image_id,
            "result": int(result),
            "result_text": result_text,
            "confidence": float(confidence),
            "probabilities": {
                "real": float(probabilities[1]),
                "ai_generated": float(probabilities[0])
            },
            "prediction": class_names[result]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=3000)