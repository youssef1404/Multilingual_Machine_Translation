from fastapi import FastAPI
from pydantic import BaseModel
from utils import translate
import uvicorn, os
from utils import load_checkpoint, load_objects
from model import EncoderRNN, AttnDecoderRNN

# Load objects
input_lang, output_lang = load_objects('input_lang', 'output_lang')
encoder = EncoderRNN(input_lang.n_words, 256).to('cpu')
decoder = AttnDecoderRNN(256, output_lang.n_words).to('cpu')
load_checkpoint(os.path.join(os.getcwd(), 'artifacts', 'checkpoint_epoch_100.pt'), encoder, decoder)

# Initialize the app 
app = FastAPI(title='Neural machine translation')
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translation: str

@app.get('/')
async def home():
    return {'Hello I am youssef kamel'}

@app.post("/translate/", response_model=TranslationResponse)
async def translate_(request: TranslationRequest):
    encoder.eval()
    decoder.eval()
    translated_sentence= translate(request.text)
    return TranslationResponse(translation=translated_sentence)

if __name__ == "__main__":
    uvicorn.run(app, port=8000)