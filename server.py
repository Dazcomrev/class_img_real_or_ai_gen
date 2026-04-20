from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
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

# Настройки БД (ИЗМЕНИТЕ!)
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
    __tablename__ = '"User"'
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

# Модель
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


# JWT utils
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
def get_history(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(401, "Authorization required")
    history = db.query(History).filter(History.UserId == current_user.UserId) \
        .order_by(History.TimeStamp.desc()).limit(50).all()
    return history


@app.post('/predict')
async def predict(
        #image_id: str = Query(...),
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

        # Save to DB only for registered users
        if current_user and not guest:
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

        return {
            #"image_id": image_id,
            "image_path": str(file_path),
            "result": result,
            "result_text": result_text,
            "confidence": float(conf),
            "probabilities": {"real": float(probs[1]), "ai_generated": float(probs[0])}
        }
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, str(e))

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)