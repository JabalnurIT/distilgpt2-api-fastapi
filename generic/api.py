from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .distilgpt2.model import Model, get_model

app = FastAPI()


class GenericRetrainRequest(BaseModel):
    texts: str

    class Config:
        from_attributes = True
        # add example value
        json_schema_extra = {
            "example": {
                "texts": "This is the first column of the first row, the second column of the first row, etc###This is the first column of the second row, the second column of the second row, etc###This is the first column of the third row, the second column of the third row, etc"
            }
        }

class GenericGenerateRequest(BaseModel):
    num_text: int

    class Config:
        from_attributes = True
        # add example value
        json_schema_extra = {
            "example": {
                "num_text": 5
            }
        }




class GenericRetrainResponse(BaseModel):
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "texts": "This is the first column of the first row, the second column of the first row, etc###This is the first column of the second row, the second column of the second row, etc###This is the first column of the third row, the second column of the third row, etc"
            }
        }

class GenericGenerateResponse(BaseModel):
    dataset: str

    class Config:
        json_schema_extra = {
            "example": {
                "texts": "This is the first column of the first row, the second column of the first row, etc###This is the first column of the second row, the second column of the second row, etc###This is the first column of the third row, the second column of the third row, etc"
            }
        }

@app.post("/retrain", response_model=GenericRetrainResponse)
async def retrain(request: GenericRetrainRequest, model: Model = Depends(get_model)):
    status = model.retrain(request.texts)
    return GenericRetrainResponse(
        status=status
    )

@app.post("/generate", response_model=GenericGenerateResponse)
async def generate(request: GenericGenerateRequest, model: Model = Depends(get_model)):
    dataset = model.generate(request.num_text)
    return GenericGenerateResponse(
        dataset=dataset
    )

    
