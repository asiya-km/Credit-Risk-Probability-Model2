from pydantic import BaseModel

class PredictRequest(BaseModel):
    # Define fields matching model features
    pass

class PredictResponse(BaseModel):
    risk_probability: float 