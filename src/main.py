from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel,Field,field_validator

base_dir=Path(__file__).resolve().parent.parent
model_dir=base_dir/"models"
models={}
class TicketRequest(BaseModel):
    text:str=Field(...,min_length=1,max_length=5000)

    @field_validator("text")
    @classmethod
    def check_text(cls,value):
        value=value.strip()
        if not value:
            raise ValueError("Ticket text cannot be empty")
        return value
class TicketResponse(BaseModel):
    category:str
    priority:str
    confidence:float

@asynccontextmanager
async def lifespan(app: FastAPI):
    category_path=model_dir/"category_model.joblib"
    priority_path=model_dir/"priority_model.joblib"
    if not category_path.exists() or not priority_path.exists():
        raise RuntimeError(
            "Model files are missing.Run train_baseline.py first"
        )

    models["category"]=joblib.load(category_path)
    models["priority"]=joblib.load(priority_path)
    yield
    models.clear()

app=FastAPI(
    title="Customer Support Ticket Classifier",
    description="Predict ticket category and priority.",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return{
        "status":"ok",
        "models_loaded":(
            "category" in models and "priority" in models
        )
    }

@app.post("/predict",response_model=TicketResponse)
def predict(request:TicketRequest):
    text=request.text
    category_model=models["category"]
    priority_model=models["priority"]
    category=category_model.predict([text])[0]
    priority=priority_model.predict([text])[0]
    category_confidence=category_model.predict_proba([text]).max()
    priority_confidence=priority_model.predict_proba([text]).max()
    confidence=(
        float(category_confidence)+float(priority_confidence)
    )/2
    return {
        "category":str(category),
        "priority":str(priority),
        "confidence":round(confidence,3)
    }