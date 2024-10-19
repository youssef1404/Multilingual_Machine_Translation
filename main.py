from fastapi import FastAPI
from pydantic import BaseModel
from utils.utils_ita import translate as translate_ita
import uvicorn, os
from utils.utils_ita import load_checkpoint, load_objects
from utils.utils_fr import predict_translation
# from utils.utils_trans import translate as translate_por

# Initialize the app 
app = FastAPI(title='Neural machine translation')
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translation: str

@app.get('/')
async def home():
    return {'Hello I am youssef kamel'}

@app.post("/translate_ita/", response_model=TranslationResponse)
async def translate1(request: TranslationRequest):
    translated_sentence= translate_ita(request.text)
    return TranslationResponse(translation=translated_sentence)

# @app.post("/translate_por/", response_model=TranslationResponse)
# async def translate2(request: TranslationRequest):
#     translated_sentence = translate_por(request.text)
#     return TranslationResponse(translation=translated_sentence)

@app.post('/translate_fra')
async def prediction(input_text :str):
    # Example usage
    # input_text = "Quel est ton nom" # French input
    translated_text = predict_translation(input_text)
    
    return {f"Translated (English): {translated_text}"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
