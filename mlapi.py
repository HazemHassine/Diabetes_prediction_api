import pickle as pkl
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

class request(BaseModel):
	pregnancies: int
	glucose: int
	skinThickness: int
	insulin: int
	bmi: float
	diabetesPedigreeFunction: int
	age: int


with open("model_pkl.pkl", "rb") as f:
	model = pkl.load(f)

@app.get('/'):
def index():
	return "<h1>This is the diabetes api</h1>"

@app.post('/predict/')
def get_prediction(req: request):
	data = req.dict()
	pregnancies = req["pregnancies"] 
	glucose = req["glucose"]
	skinThickness = req["skinThickness"]
	insulin = req["insulin"]
	bmi = req["bmi"]
	diabetesPedigreeFunction = req["diabetesPedigreeFunction"]
	age = req["age"]
	return {'prediction': model.predict([[pregnancies,glucose,skinThickness,insulin,bmi,diabetesPedigreeFunction,age]])}