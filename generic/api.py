from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .distilgpt2.model import Model, get_model

app = FastAPI()


class NaturalnessRequest(BaseModel):
    texts: str

    class Config:
        orm_mode = True
        # add example value
        schema_extra = {
            "example": {
                "texts": "This is the first column of the first row, the second column of the first row, etc###This is the first column of the second row, the second column of the second row, etc###This is the first column of the third row, the second column of the third row, etc"
            }
        }





class NaturalnessRetrainResponse(BaseModel):
    status: str

    class Config:
        schema_extra = {
            "example": {
                "texts": "This is the first column of the first row, the second column of the first row, etc###This is the first column of the second row, the second column of the second row, etc###This is the first column of the third row, the second column of the third row, etc"
            }
        }

class NaturalnessGenerateResponse(BaseModel):
    dataset: str

    class Config:
        schema_extra = {
            "example": {
                "texts": "This is the first column of the first row, the second column of the first row, etc###This is the first column of the second row, the second column of the second row, etc###This is the first column of the third row, the second column of the third row, etc"
            }
        }



@app.post("/generate", response_model=NaturalnessGenerateResponse)
async def generate(request: NaturalnessRequest, model: Model = Depends(get_model)):
    dataset = model.generate(request.texts)
    return NaturalnessGenerateResponse(
        dataset=dataset
    )

    

@app.post("/retrain", response_model=NaturalnessRetrainResponse)
async def retrain(request: NaturalnessRequest, model: Model = Depends(get_model)):
    status = model.retrain(request.texts)
    return NaturalnessRetrainResponse(
        status=status
    )