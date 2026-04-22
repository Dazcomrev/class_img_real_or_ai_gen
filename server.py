from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func, delete
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

def cleanup_old_history():
    while True:
        try:
            with SessionLocal() as db:
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)#minutes=2
                old_items = (
                    db.query(History)
                    .filter(History.TimeStamp < cutoff)
                    .all()
                )

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

# Запуск фонового таймера
cleanup_thread = threading.Thread(target=cleanup_old_history, daemon=True)
cleanup_thread.start()

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


# Модели БД
class User(Base):
    __tablename__ = 'User'
    UserId = Column(Integer, primary_key=True, index=True)
    Name = Column(String(100), nullable=False)
    Email = Column(String(255), unique=True, index=True, nullable=False)
    PasswordHash = Column(String(255), nullable=False)
    EmailVerified = Column(Boolean, default=False)
    CreatedAt = Column(DateTime(timezone=True), server_default=func.now())
    UpdatedAt = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


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


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Модель (без изменений)
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


# Pydantic
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


# JWT utils (без изменений)
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


@app.get("/")
async def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Create static/index.html"}


@app.get('/status')
def status():
    model_status = "Загружена" if model else "Ошибка загрузки"
    return {"status": "Server OK", "model": model_status}


# Регистрация (без изменений)
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


# Логин (без изменений)
@app.post('/token', response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.Email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.PasswordHash):
        raise HTTPException(401, "Invalid email/password")
    token = create_access_token({"sub": str(user.UserId)})
    return {"access_token": token, "token_type": "bearer"}


# НОВЫЕ ЭНДПОИНТЫ

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

    print(f"FOUND {len(history)} records for user {current_user.UserId}")
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

    # Удаляем файл изображения
    image_path = Path(history.ImagePath)
    if image_path.exists():
        image_path.unlink()

    # Удаляем запись из БД
    db.delete(history)
    db.commit()

    return {"message": "History item deleted"}


@app.delete('/history/all/clear')
def clear_history(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    print(f"Current user ID: {current_user.UserId}")
    print(f"History records for user {current_user.UserId}:")
    q = db.query(History).filter(History.UserId == current_user.UserId)
    print([h.HistoryId for h in q.all()])

    user_history = q.all()

    for history in user_history:
        image_path = Path(history.ImagePath)
        if image_path.exists():
            image_path.unlink()

    deleted_count = db.query(History).filter(History.UserId == current_user.UserId).delete()
    db.commit()
    print(f"Deleted {deleted_count} records")

    return {"message": "History cleared", "deleted_count": deleted_count}


@app.delete('/account')
def delete_account(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    # Удаляем историю и файлы
    user_history = db.query(History).filter(History.UserId == current_user.UserId).all()
    for history in user_history:
        image_path = Path(history.ImagePath)
        if image_path.exists():
            image_path.unlink()

    # Удаляем историю из БД
    db.query(History).filter(History.UserId == current_user.UserId).delete()

    # Удаляем аккаунт
    db.delete(current_user)
    db.commit()

    return {"message": "Account and history deleted"}


# Predict (без изменений)
@app.post('/predict')
async def predict(
        image_id: Optional[str] = Query(None),
        guest: bool = Query(False),
        file: UploadFile = File(...),
        current_user=Depends(get_current_user),
        db: Session = Depends(get_db)):
    if not model:
        raise HTTPException(500, "Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Images only")

    prefix = f"{current_user.UserId}_" if current_user else "guest_"
    filename = f"{prefix}{uuid.uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / filename
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        image = Image.open(file_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred = torch.max(probs, 0)
            result = pred.item()

        result_text = "Реальное изображение" if result == 1 else "Изображение сгенерировано нейронной сетью"

        if current_user and not guest:
            print(f"SAVING to DB for user {current_user.UserId}")
            history = History(
                UserId=current_user.UserId,
                ImagePath=str(file_path),
                Prediction=result,
                Confidence=float(conf),
                ProbReal=float(probs[1]),
                ProbAI=float(probs[0])
            )
            db.add(history)
            db.commit()
            db.refresh(history)
            print(f"SAVED! HistoryId: {history.HistoryId}")

        return {
            "image_path": str(file_path),
            "result": result,
            "result_text": result_text,
            "confidence": float(conf),
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


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)