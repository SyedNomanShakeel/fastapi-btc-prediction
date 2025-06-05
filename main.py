from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
import joblib

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# SQLAlchemy User model
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

# Pydantic schemas
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    class Config:
        orm_mode = True

class PredictionInput(BaseModel):
    volume_btc: float
    close_eth: float
    volume_eth: float
    close_usdt: float
    volume_usdt: float
    close_bnb: float
    volume_bnb: float

class PredictionOut(PredictionInput):
    predicted_btc: float

# Load ML models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_random_model.pkl")
SCALER_MODEL_PATH = os.path.join(BASE_DIR, "scaler_model.pkl")

try:
    rf_model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_MODEL_PATH)
except Exception as e:
    rf_model = None
    scaler = None
    print(f"Model loading error: {e}")

def predict_btc(features):
    if rf_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    scaled = scaler.transform([features])
    return rf_model.predict(scaled)[0]

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the BTC Prediction API"}

@app.post("/signup", response_model=UserOut)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    fake_hashed_password = "hashed_" + user.password
    new_user = User(username=user.username, email=user.email, hashed_password=fake_hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or db_user.hashed_password != "hashed_" + user.password:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    return {"message": "Login successful", "user_id": db_user.id, "username": db_user.username}

@app.post("/predict", response_model=PredictionOut)
def predict(input: PredictionInput):
    features = [
        input.volume_btc,
        input.close_eth,
        input.volume_eth,
        input.close_usdt,
        input.volume_usdt,
        input.close_bnb,
        input.volume_bnb,
    ]
    predicted_btc = predict_btc(features)
    return PredictionOut(**input.dict(), predicted_btc=predicted_btc)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
