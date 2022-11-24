import pickle as pkl
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
app = FastAPI()

class request(BaseModel):
	pregnancies: int
	bloodpressure: int
	glucose: int
	skinThickness: int
	insulin: int
	bmi: float
	diabetesPedigreeFunction: float
	age: int


with open("model_pkl.pkl", "rb") as f:
	model = pkl.load(f)
data = pd.read_csv("data.csv")

with open("StandardScaler_pkl.pkl", "rb") as f:
	sc_X = pkl.load(f)

@app.get('/')
def index():
	return "This is the diabetes api The request should be in json form containg these fields:pregnancies: int/bloodpressure: int/glucose: in/skinThickness: int/insulin: int/bmi: float/diabetesPedigreeFunction: int/age: int"

@app.post('/predict/')
async def get_prediction(req: request):
	data = req.dict()
	pregnancies = data["pregnancies"] 
	glucose = data["glucose"]
	bloodpressure = data["bloodpressure"]
	skinThickness = data["skinThickness"]
	insulin = data["insulin"]
	bmi = data["bmi"]
	diabetesPedigreeFunction = data["diabetesPedigreeFunction"]
	age = data["age"]
	return {"pred": str(model.predict(sc_X.transform([[pregnancies,glucose,bloodpressure,skinThickness,insulin,bmi,diabetesPedigreeFunction,age]]))[0])}