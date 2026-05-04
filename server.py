from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import uuid
import shutil
from pathlib import Path
import uvicorn
import threading
import time
from datetime import datetime, timedelta, timezone


# Фоновая очистка истории
def cleanup_old_history():
    while True:
        try:
            with SessionLocal() as db:
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                old_items = db.query(History).filter(History.TimeStamp < cutoff).all()

                deleted_count = 0
                for item in old_items:
                    image_path = Path(item.ImagePath)
                    if image_path.exists():
                        image_path.unlink()
                    db.delete(item)
                    deleted_count += 1

                if deleted_count > 0:
                    db.commit()
                    print(f"Cleared {deleted_count} old items")
        except Exception as e:
            print(f"Cleanup error: {e}")
        time.sleep(60 * 60)  # раз в час


# Настройки БД
DATABASE_URL = "postgresql://postgres:123456@localhost/ClassImgRealOrGenAI"
SECRET_KEY = "super-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="Image Real/AI Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# База данных
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Модели БД (оптимизированные)
class User(Base):
    __tablename__ = 'User'
    UserId = Column(Integer, primary_key=True, index=True)
    Name = Column(String(100), nullable=False)
    Email = Column(String(255), unique=True, index=True, nullable=False)
    PasswordHash = Column(String(255), nullable=False)
    CreatedAt = Column(DateTime(timezone=True), server_default=func.now())


class History(Base):
    __tablename__ = "History"
    HistoryId = Column(Integer, primary_key=True, index=True)
    UserId = Column(Integer, nullable=False)
    ImagePath = Column(String(500), nullable=False)
    Prediction = Column(Integer, nullable=False)
    Confidence = Column(Float, nullable=False)
    ProbReal = Column(Float, nullable=False)
    ProbAI = Column(Float, nullable=False)
    TimeStamp = Column(DateTime(timezone=True), server_default=func.now())


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Запуск фоновой очистки
cleanup_thread = threading.Thread(target=cleanup_old_history, daemon=True)
cleanup_thread.start()

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Модель машинного обучения
model_path = "best_resnet_last_two_layers_cpu.pkl"
model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

try:
    state_dict = torch.load(model_path, map_location='cpu')
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(state_dict)
    model.eval()
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Pydantic модели
class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class HistoryResponse(BaseModel):
    HistoryId: int
    ImagePath: str
    Prediction: int
    Confidence: float
    ProbReal: float
    ProbAI: float
    TimeStamp: datetime


# JWT утилиты
def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        user = db.query(User).filter(User.UserId == user_id).first()
        return user
    except JWTError:
        return None


# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# Эндпоинты
@app.get("/")
async def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Create static/index.html"}


@app.get('/status')
def status():
    model_status = "Загружена" if model else "Ошибка загрузки"
    return {"status": "Server OK", "model": model_status}


@app.post('/register', status_code=201)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.Email == user_data.email).first():
        raise HTTPException(400, "Email already registered")
    hashed = get_password_hash(user_data.password)
    user = User(Name=user_data.name, Email=user_data.email, PasswordHash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Добавлен пользователь", "user_id": user.UserId}


@app.post('/token', response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.Email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.PasswordHash):
        raise HTTPException(401, "Invalid email/password")
    token = create_access_token({"sub": str(user.UserId)})
    return {"access_token": token, "token_type": "bearer"}


@app.get('/history', response_model=list[HistoryResponse])
def get_history(
        sort: str = Query("desc", regex="^(asc|desc)$"),
        current_user=Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    query = db.query(History).filter(History.UserId == current_user.UserId)

    if sort == "asc":
        history = query.order_by(History.TimeStamp.asc()).all()
    else:
        history = query.order_by(History.TimeStamp.desc()).all()

    return history


@app.delete('/history/one/{history_id}')
def delete_history(history_id: int, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    history = db.query(History).filter(
        History.HistoryId == history_id,
        History.UserId == current_user.UserId
    ).first()

    if not history:
        raise HTTPException(404, "History item not found")

    image_path = Path(history.ImagePath)
    if image_path.exists():
        image_path.unlink()

    db.delete(history)
    db.commit()

    return {"message": "History item deleted"}


@app.delete('/history/all/clear')
def clear_history(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    user_history = db.query(History).filter(History.UserId == current_user.UserId).all()

    for history in user_history:
        image_path = Path(history.ImagePath)
        if image_path.exists():
            image_path.unlink()

    deleted_count = db.query(History).filter(History.UserId == current_user.UserId).delete()
    db.commit()

    return {"message": "History cleared", "deleted_count": deleted_count}


@app.delete('/account')
def delete_account(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    user_history = db.query(History).filter(History.UserId == current_user.UserId).all()
    for history in user_history:
        image_path = Path(history.ImagePath)
        if image_path.exists():
            image_path.unlink()

    db.query(History).filter(History.UserId == current_user.UserId).delete()
    db.delete(current_user)
    db.commit()

    return {"message": "Account and history deleted"}


@app.post('/predict')
async def predict(
        guest: bool = Query(False),
        file: UploadFile = File(...),
        current_user=Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if not model:
        raise HTTPException(500, "Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Images only")

    # Генерация имени файла
    prefix = f"{current_user.UserId}_" if current_user else "guest_"
    filename = f"{prefix}{uuid.uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    # Сохранение файла
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        # Классификация изображения
        image = Image.open(file_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, prediction = torch.max(probs, 0)
            result = prediction.item()

        result_text = "Реальное изображение" if result == 1 else "Изображение сгенерировано нейронной сетью"

        # Сохранение в историю только для авторизованных пользователей (не гостей)
        if current_user and not guest:
            history = History(
                UserId=current_user.UserId,
                ImagePath=str(file_path),
                Prediction=result,
                Confidence=float(confidence),
                ProbReal=float(probs[1]),
                ProbAI=float(probs[0])
            )
            db.add(history)
            db.commit()

        return {
            "image_path": str(file_path),
            "result": result,
            "result_text": result_text,
            "confidence": float(confidence),
            "probabilities": {
                "real": float(probs[1]),
                "ai_generated": float(probs[0])
            }
        }

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class UserInfoResponse(BaseModel):
    user_id: int
    name: str
    email: str


class ChangeNameRequest(BaseModel):
    name: str


@app.get('/user/info', response_model=UserInfoResponse)
def get_user_info(current_user=Depends(get_current_user)):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    return {
        "user_id": current_user.UserId,
        "name": current_user.Name,
        "email": current_user.Email
    }


@app.put('/user/change-name')
def change_user_name(
        request: ChangeNameRequest,
        current_user=Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    if not request.name or len(request.name.strip()) < 2:
        raise HTTPException(400, "Name must be at least 2 characters long")

    current_user.Name = request.name.strip()
    db.commit()
    db.refresh(current_user)

    return {"message": "Name updated successfully", "name": current_user.Name}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)