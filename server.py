from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
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
import re
import dns.resolver

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


# Модели БД (оптимизированные) - ОПРЕДЕЛЯЕМ ВСЕ МОДЕЛИ ДО ИХ ИСПОЛЬЗОВАНИЯ
class User(Base):
    __tablename__ = 'User'
    UserId = Column(Integer, primary_key=True, index=True)
    Name = Column(String(100), nullable=False)
    Email = Column(String(255), unique=True, index=True, nullable=False)
    PasswordHash = Column(String(255), nullable=False)
    CreatedAt = Column(DateTime(timezone=True), server_default=func.now())

    # Статистические поля
    TotalChecks = Column(Integer, default=0)
    TotalRealChecks = Column(Integer, default=0)
    TotalAIChecks = Column(Integer, default=0)
    TotalFeedback = Column(Integer, default=0)
    TotalTrueReal = Column(Integer, default=0)
    TotalTrueAI = Column(Integer, default=0)
    TotalFalseReal = Column(Integer, default=0)
    TotalFalseAI = Column(Integer, default=0)
    TotalFavorites = Column(Integer, default=0)
    LastActiveAt = Column(DateTime(timezone=True), nullable=True)
    AccuracyScore = Column(Float, default=0)


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


class Favorite(Base):
    __tablename__ = "Favorite"
    FavoriteId = Column(Integer, primary_key=True, index=True)
    UserId = Column(Integer, nullable=False)
    HistoryId = Column(Integer, nullable=False, unique=True)
    AddedAt = Column(DateTime(timezone=True), server_default=func.now())


class Feedback(Base):
    __tablename__ = "Feedback"
    FeedbackId = Column(Integer, primary_key=True, index=True)
    UserId = Column(Integer, nullable=False)
    HistoryId = Column(Integer, nullable=False)
    ImagePath = Column(String(500), nullable=False)
    FeedbackImagePath = Column(String(500), nullable=True)
    OriginalPrediction = Column(Integer, nullable=False)
    OriginalProbReal = Column(Float, nullable=False)
    OriginalProbAI = Column(Float, nullable=False)
    UserCorrection = Column(Integer, nullable=False)
    FeedbackType = Column(String(20), nullable=False)
    CreatedAt = Column(DateTime(timezone=True), server_default=func.now())


# Создаем таблицы
Base.metadata.create_all(bind=engine)


# Функция для обновления статистики пользователя (определяем ДО cleanup_old_history)
def update_user_stats(user_id: int, db: Session):
    """Пересчитывает и обновляет статистику пользователя"""
    user = db.query(User).filter(User.UserId == user_id).first()
    if not user:
        return

    # Общая статистика из истории
    total_checks = db.query(History).filter(History.UserId == user_id).count()
    total_real = db.query(History).filter(
        History.UserId == user_id,
        History.Prediction == 1
    ).count()
    total_ai = total_checks - total_real

    # Статистика обратной связи
    total_feedback = db.query(Feedback).filter(Feedback.UserId == user_id).count()
    total_true_real = db.query(Feedback).filter(
        Feedback.UserId == user_id,
        Feedback.FeedbackType == "TrueReal"
    ).count()
    total_true_ai = db.query(Feedback).filter(
        Feedback.UserId == user_id,
        Feedback.FeedbackType == "TrueAI"
    ).count()
    total_false_real = db.query(Feedback).filter(
        Feedback.UserId == user_id,
        Feedback.FeedbackType == "FalseReal"
    ).count()
    total_false_ai = db.query(Feedback).filter(
        Feedback.UserId == user_id,
        Feedback.FeedbackType == "FalseAI"
    ).count()

    # Статистика избранного
    total_favorites = db.query(Favorite).filter(Favorite.UserId == user_id).count()

    # Последняя активность (исправляем проблему с timezone)
    last_history = db.query(History.TimeStamp).filter(
        History.UserId == user_id
    ).order_by(History.TimeStamp.desc()).first()

    last_feedback = db.query(Feedback.CreatedAt).filter(
        Feedback.UserId == user_id
    ).order_by(Feedback.CreatedAt.desc()).first()

    # Исправляем сравнение datetime с timezone
    min_datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)

    history_time = last_history[0] if last_history else min_datetime
    feedback_time = last_feedback[0] if last_feedback else min_datetime

    # Приводим оба datetime к timezone-aware
    if history_time.tzinfo is None:
        history_time = history_time.replace(tzinfo=timezone.utc)
    if feedback_time.tzinfo is None:
        feedback_time = feedback_time.replace(tzinfo=timezone.utc)

    last_active = max(history_time, feedback_time)

    # Точность модели
    total_evaluated = total_true_real + total_true_ai + total_false_real + total_false_ai
    accuracy = ((total_true_real + total_true_ai) / total_evaluated * 100) if total_evaluated > 0 else 0

    # Обновляем пользователя
    user.TotalChecks = total_checks
    user.TotalRealChecks = total_real
    user.TotalAIChecks = total_ai
    user.TotalFeedback = total_feedback
    user.TotalTrueReal = total_true_real
    user.TotalTrueAI = total_true_ai
    user.TotalFalseReal = total_false_real
    user.TotalFalseAI = total_false_ai
    user.TotalFavorites = total_favorites
    user.LastActiveAt = last_active if last_active != min_datetime else None
    user.AccuracyScore = round(accuracy, 2)

    db.commit()


# Фоновая очистка истории (использует модель Favorite, которая уже определена)
def cleanup_old_history():
    while True:
        try:
            with SessionLocal() as db:
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)

                # Находим элементы для удаления, которые НЕ в избранном
                old_items = db.query(History).filter(
                    History.TimeStamp < cutoff,
                    ~History.HistoryId.in_(db.query(Favorite.HistoryId))
                ).all()

                deleted_count = 0
                for item in old_items:
                    image_path = Path(item.ImagePath)
                    if image_path.exists():
                        image_path.unlink()
                    db.delete(item)
                    deleted_count += 1

                if deleted_count > 0:
                    db.commit()
                    print(f"Cleared {deleted_count} old items (favorites preserved)")
        except Exception as e:
            print(f"Cleanup error: {e}")
        time.sleep(60 * 60)  # раз в час


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
#model_path = "best_resnet_last_two_layers_cpu.pkl"
model_path = "absolute_finale.pkl"
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

# Создайте папки для обратной связи
FEEDBACK_DIR = Path("feedback_images")
FEEDBACK_REAL_DIR = FEEDBACK_DIR / "real"
FEEDBACK_FAKE_DIR = FEEDBACK_DIR / "fake"
FEEDBACK_REAL_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FAKE_DIR.mkdir(parents=True, exist_ok=True)


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


class FavoriteResponse(BaseModel):
    FavoriteId: int
    HistoryId: int
    AddedAt: datetime
    history_item: HistoryResponse


class FeedbackRequest(BaseModel):
    history_id: int
    user_correction: int


class UserInfoResponse(BaseModel):
    user_id: int
    name: str
    email: str


class ChangeNameRequest(BaseModel):
    name: str


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

def validate_email(email: str) -> bool:
    # Базовая проверка формата
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False

    # Проверка MX-записи домена (опционально)
    try:
        domain = email.split('@')[1]
        dns.resolver.resolve(domain, 'MX')
        return True
    except:
        return False  # Домен не имеет MX записи

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
    #model_status = "Загружена" if model else "Ошибка загрузки"
    model_status = "Uploaded" if model else "Loading error"
    return {"status": "Server OK", "model": model_status}


@app.post('/register', status_code=201)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    if not validate_email(user_data.email):
        raise HTTPException(400, "Invalid email format or domain doesn't exist")
    if db.query(User).filter(User.Email == user_data.email).first():
        raise HTTPException(400, "Email already registered")
    hashed = get_password_hash(user_data.password)
    user = User(Name=user_data.name, Email=user_data.email, PasswordHash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    #return {"message": "Добавлен пользователь", "user_id": user.UserId}
    return {"message": "User added", "user_id": user.UserId}


@app.post('/token', response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.Email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.PasswordHash):
        raise HTTPException(401, "Invalid email/password")
    token = create_access_token({"sub": str(user.UserId)})
    return {"access_token": token, "token_type": "bearer"}


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

    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.ico'}
    ALLOWED_MIMETYPES = {
        'image/jpeg', 'image/png', 'image/bmp', 'image/webp',
        'image/tiff', 'image/x-icon', 'image/vnd.microsoft.icon'
    }

    # Проверка
    if file.content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(400, f"Unsupported image type. Supported: {', '.join(ALLOWED_MIMETYPES)}")

    # Для гостей - не сохраняем файл, работаем в памяти
    if guest or not current_user:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                #confidence, prediction = torch.max(probs, 0)
                _, prediction = torch.max(probs, 0)
                result = prediction.item()

            confidence = abs(float(probs[1]) - float(probs[0]))

            return {
                "result": result,
                "result_text": "Реальное изображение" if result == 1 else "Изображение сгенерировано нейронной сетью",
                "confidence": confidence, #float(confidence)
                "probabilities": {
                    "real": float(probs[1]),
                    "ai_generated": float(probs[0])
                },
                "guest_mode": True
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    # Для авторизованных пользователей - сохраняем
    filename = f"{current_user.UserId}_{uuid.uuid4()}_{file.filename}"
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
            #confidence, prediction = torch.max(probs, 0)
            _, prediction = torch.max(probs, 0)
            result = prediction.item()

        confidence = abs(float(probs[1]) - float(probs[0]))

        history = History(
            UserId=current_user.UserId,
            ImagePath=str(file_path),
            Prediction=result,
            Confidence=confidence,#float(confidence),
            ProbReal=float(probs[1]),
            ProbAI=float(probs[0])
        )
        db.add(history)
        db.commit()

        # Обновляем статистику пользователя
        update_user_stats(current_user.UserId, db)

        return {
            "image_path": str(file_path),
            "result": result,
            "result_text": "Реальное изображение" if result == 1 else "Изображение сгенерировано нейронной сетью",
            "confidence": confidence, #float(confidence)
            "probabilities": {
                "real": float(probs[1]),
                "ai_generated": float(probs[0])
            },
            "history_id": history.HistoryId,
            "guest_mode": False
        }

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


'''def adjust_timestamp_to_30days_ago(timestamp: datetime) -> datetime:
    """Сдвигает временную метку на 30 дней назад"""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp - timedelta(days=30)'''

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

    '''# Модифицируем каждый элемент истории
    modified_history = []
    for item in history:
        # Создаем словарь с данными
        item_dict = {
            "HistoryId": item.HistoryId,
            "ImagePath": item.ImagePath,
            "Prediction": item.Prediction,
            "Confidence": item.Confidence,
            "ProbReal": item.ProbReal,
            "ProbAI": item.ProbAI,
            "TimeStamp": adjust_timestamp_to_30days_ago(item.TimeStamp)
        }
        modified_history.append(item_dict)'''

    return history # modified_history #


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

    # Обновляем статистику пользователя
    update_user_stats(current_user.UserId, db)

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

    # Обновляем статистику пользователя
    update_user_stats(current_user.UserId, db)

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

    # Удаляем записи из БД, НО НЕ УДАЛЯЕМ ФАЙЛЫ feedback_images
    # Просто удаляем записи, файлы остаются для обучения модели
    db.query(Favorite).filter(Favorite.UserId == current_user.UserId).delete()
    db.query(History).filter(History.UserId == current_user.UserId).delete()

    # Опционально: можно оставить Feedback для статистики, но без привязки к пользователю
    # Либо удалить записи, но оставить файлы
    db.query(Feedback).filter(Feedback.UserId == current_user.UserId).delete()

    db.delete(current_user)
    db.commit()

    return {"message": "Account and all data deleted"}


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


@app.post('/favorites/{history_id}')
def add_to_favorites(
        history_id: int,
        current_user=Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    history_item = db.query(History).filter(
        History.HistoryId == history_id,
        History.UserId == current_user.UserId
    ).first()

    if not history_item:
        raise HTTPException(404, "History item not found")

    existing_favorite = db.query(Favorite).filter(
        Favorite.UserId == current_user.UserId,
        Favorite.HistoryId == history_id
    ).first()

    if existing_favorite:
        raise HTTPException(400, "Item already in favorites")

    favorite = Favorite(
        UserId=current_user.UserId,
        HistoryId=history_id
    )
    db.add(favorite)
    db.commit()
    db.refresh(favorite)

    # Обновляем статистику пользователя
    update_user_stats(current_user.UserId, db)

    return {"message": "Added to favorites", "favorite_id": favorite.FavoriteId}


@app.delete('/favorites/{history_id}')
def remove_from_favorites(
        history_id: int,
        current_user=Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    favorite = db.query(Favorite).filter(
        Favorite.UserId == current_user.UserId,
        Favorite.HistoryId == history_id
    ).first()

    if not favorite:
        raise HTTPException(404, "Item not in favorites")

    db.delete(favorite)
    db.commit()

    # Обновляем статистику пользователя
    update_user_stats(current_user.UserId, db)

    return {"message": "Removed from favorites"}


@app.get('/favorites', response_model=list[FavoriteResponse])
def get_favorites(
        current_user=Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    favorites = db.query(Favorite).filter(
        Favorite.UserId == current_user.UserId
    ).order_by(Favorite.AddedAt.desc()).all()

    result = []
    for fav in favorites:
        history_item = db.query(History).filter(History.HistoryId == fav.HistoryId).first()
        if history_item:
            result.append({
                "FavoriteId": fav.FavoriteId,
                "HistoryId": fav.HistoryId,  # Убедимся, что имя поля правильное
                "AddedAt": fav.AddedAt, #adjust_timestamp_to_30days_ago(fav.AddedAt), #
                "history_item": {
                    "HistoryId": history_item.HistoryId,
                    "ImagePath": history_item.ImagePath,
                    "Prediction": history_item.Prediction,
                    "Confidence": history_item.Confidence,
                    "ProbReal": history_item.ProbReal,
                    "ProbAI": history_item.ProbAI,
                    "TimeStamp": history_item.TimeStamp #adjust_timestamp_to_30days_ago(history_item.TimeStamp), #
                }
            })

    return result


@app.post('/feedback')
def submit_feedback(
        feedback_data: FeedbackRequest,
        current_user=Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    history_item = db.query(History).filter(
        History.HistoryId == feedback_data.history_id,
        History.UserId == current_user.UserId
    ).first()

    if not history_item:
        raise HTTPException(404, "History item not found")

    existing_feedback = db.query(Feedback).filter(
        Feedback.UserId == current_user.UserId,
        Feedback.HistoryId == feedback_data.history_id
    ).first()

    if existing_feedback:
        raise HTTPException(400, "Feedback already submitted for this image")

    original_pred = history_item.Prediction
    user_correction = feedback_data.user_correction

    # Определяем тип обратной связи
    # user_correction уже приходит от клиента (0 - AI, 1 - Real)
    if original_pred == 1 and user_correction == 1:
        feedback_type = "TrueReal"
    elif original_pred == 0 and user_correction == 0:
        feedback_type = "TrueAI"
    elif original_pred == 1 and user_correction == 0:
        feedback_type = "FalseReal"
    else:  # original_pred == 0 and user_correction == 1
        feedback_type = "FalseAI"

    original_path = Path(history_item.ImagePath)
    feedback_image_path = None

    # Сохраняем копию изображения в соответствующую папку (ВСЕГДА сохраняем, даже если пользователь удалит аккаунт)
    if original_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_prefix = f"user_{current_user.UserId}"
        original_filename = original_path.name
        new_filename = f"{timestamp}_{user_prefix}_{original_filename}"

        # Сохраняем в папку в зависимости от того, что пользователь указал как правильный ответ
        if user_correction == 1:  # Пользователь считает, что реальное
            dest_dir = FEEDBACK_REAL_DIR
        else:  # Пользователь считает, что AI
            dest_dir = FEEDBACK_FAKE_DIR

        dest_path = dest_dir / new_filename
        shutil.copy2(original_path, dest_path)
        feedback_image_path = str(dest_path)

    # Сохраняем обратную связь
    feedback = Feedback(
        UserId=current_user.UserId,
        HistoryId=feedback_data.history_id,
        ImagePath=str(original_path),
        FeedbackImagePath=feedback_image_path,
        OriginalPrediction=original_pred,
        OriginalProbReal=history_item.ProbReal,
        OriginalProbAI=history_item.ProbAI,
        UserCorrection=user_correction,
        FeedbackType=feedback_type
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    # Обновляем статистику пользователя
    update_user_stats(current_user.UserId, db)

    return {
        "message": "Feedback submitted successfully",
        "feedback_type": feedback_type,
        "feedback_id": feedback.FeedbackId
    }


@app.get('/feedback/my')
def get_my_feedbacks(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    feedbacks = db.query(Feedback).filter(
        Feedback.UserId == current_user.UserId
    ).all()

    return [{"history_id": fb.HistoryId} for fb in feedbacks]

'''@app.get('/feedback/my')
def get_my_feedback(
        current_user=Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    feedbacks = db.query(Feedback).filter(
        Feedback.UserId == current_user.UserId
    ).order_by(Feedback.CreatedAt.desc()).all()

    return [
        {
            "feedback_id": f.FeedbackId,
            "history_id": f.HistoryId,
            "original_prediction": f.OriginalPrediction,
            "user_correction": f.UserCorrection,
            "feedback_type": f.FeedbackType,
            "created_at": f.CreatedAt,
            "image_path": f.ImagePath,
            "feedback_image_path": f.FeedbackImagePath
        }
        for f in feedbacks
    ]'''


@app.get('/stats/my')
def get_my_stats(
        current_user=Depends(get_current_user),
        db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(401, "Authorization required")

    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    daily_activity = db.query(
        func.date(History.TimeStamp).label('date'),
        func.count(History.HistoryId).label('count')
    ).filter(
        History.UserId == current_user.UserId,
        History.TimeStamp >= thirty_days_ago
    ).group_by(func.date(History.TimeStamp)).order_by(func.date(History.TimeStamp)).all()

    daily_checks = [{"date": str(day.date), "count": day.count} for day in daily_activity]
    '''# Сдвигаем даты на 30 дней назад
    daily_checks = []
    for day in daily_activity:
        # Преобразуем строку даты в datetime объект
        original_date = datetime.strptime(str(day.date), '%Y-%m-%d')
        # Сдвигаем на 30 дней назад
        shifted_date = original_date - timedelta(days=30)
        daily_checks.append({
            "date": shifted_date.strftime('%Y-%m-%d'),
            "count": day.count
        })'''

    confidence_distribution = {
        "very_low": db.query(History).filter(
            History.UserId == current_user.UserId,
            History.Confidence < 0.6
        ).count(),
        "low": db.query(History).filter(
            History.UserId == current_user.UserId,
            History.Confidence >= 0.6,
            History.Confidence < 0.75
        ).count(),
        "medium": db.query(History).filter(
            History.UserId == current_user.UserId,
            History.Confidence >= 0.75,
            History.Confidence < 0.9
        ).count(),
        "high": db.query(History).filter(
            History.UserId == current_user.UserId,
            History.Confidence >= 0.9
        ).count()
    }

    '''# Корректируем LastActiveAt, если он существует
    last_active_adjusted = None
    if current_user.LastActiveAt:
        last_active_adjusted = adjust_timestamp_to_30days_ago(current_user.LastActiveAt).isoformat()'''

    return {
        "total_checks": current_user.TotalChecks,
        "real_count": current_user.TotalRealChecks,
        "ai_count": current_user.TotalAIChecks,
        "real_percentage": round(current_user.TotalRealChecks / current_user.TotalChecks * 100,
                                 2) if current_user.TotalChecks > 0 else 0,
        "ai_percentage": round(current_user.TotalAIChecks / current_user.TotalChecks * 100,
                               2) if current_user.TotalChecks > 0 else 0,
        "total_feedback": current_user.TotalFeedback,
        "true_real": current_user.TotalTrueReal,
        "true_ai": current_user.TotalTrueAI,
        "false_real": current_user.TotalFalseReal,
        "false_ai": current_user.TotalFalseAI,
        "model_accuracy": current_user.AccuracyScore,
        "favorites_count": current_user.TotalFavorites,
        "last_active": current_user.LastActiveAt.isoformat() if current_user.LastActiveAt else None, #last_active_adjusted, #
        "daily_activity": daily_checks,
        "confidence_distribution": confidence_distribution,
        "avg_confidence_real": round(db.query(func.avg(History.Confidence)).filter(
            History.UserId == current_user.UserId,
            History.Prediction == 1
        ).scalar() or 0, 2),
        "avg_confidence_ai": round(db.query(func.avg(History.Confidence)).filter(
            History.UserId == current_user.UserId,
            History.Prediction == 0
        ).scalar() or 0, 2)
    }


'''@app.get('/stats/leaderboard')
def get_leaderboard(
        limit: int = Query(10, ge=1, le=50),
        db: Session = Depends(get_db)
):
    top_by_accuracy = db.query(User).filter(
        User.TotalFeedback >= 5
    ).order_by(User.AccuracyScore.desc()).limit(limit).all()

    top_by_activity = db.query(User).order_by(
        User.TotalChecks.desc()
    ).limit(limit).all()

    return {
        "top_by_accuracy": [
            {
                "name": user.Name,
                "accuracy": user.AccuracyScore,
                "feedback_count": user.TotalFeedback
            }
            for user in top_by_accuracy
        ],
        "top_by_activity": [
            {
                "name": user.Name,
                "total_checks": user.TotalChecks
            }
            for user in top_by_activity
        ]
    }'''


# ВРЕМЕННЫЙ скрипт для очистки старых гостевых файлов

'''def cleanup_legacy_guest_files():
    """Одноразовая очистка старых гостевых файлов"""
    with SessionLocal() as db:
        deleted_count = 0
        if UPLOAD_DIR.exists():
            for file_path in UPLOAD_DIR.iterdir():
                if file_path.is_file():
                    # Проверяем, принадлежит ли файл существующему пользователю
                    # Формат: {user_id}_...
                    parts = file_path.stem.split('_')
                    if len(parts) > 0 and parts[0].isdigit():
                        user_id = int(parts[0])
                        user_exists = db.query(User).filter(User.UserId == user_id).first()
                        if not user_exists:
                            file_path.unlink()
                            deleted_count += 1
                            print(f"Deleted orphan file: {file_path}")
                    else:
                        # Файл без user_id (гостевой) - удаляем
                        file_path.unlink()
                        deleted_count += 1
                        print(f"Deleted guest file: {file_path}")

        print(f"Cleaned up {deleted_count} legacy guest files")'''

# cleanup_legacy_guest_files()

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)