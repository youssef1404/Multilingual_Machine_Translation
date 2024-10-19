import os 
import joblib
import numpy as np
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

MAX_SEQUENCE_LENGTH_SRC = 58
MAX_SEQUENCE_LENGTH_IN_TRG = 56

# load src_tokenizer & trg_tokenizer
tokenizer_src = joblib.load(os.path.join("artifacts","tokenizer_src.pkl"))
tokenizer_trg = joblib.load(os.path.join("artifacts","tokenizer_trg.pkl"))


# Model
model = load_model(os.path.join('artifacts', 'best_model (5).keras'))


def src_preprocessing(text: str) -> str:
    
    new_text = re.sub(r'[,\.!?»«"]', '', text)

    
    return new_text

def predict_translation(input_sequence, model=model, tokenizer_src=tokenizer_src, tokenizer_trg=tokenizer_trg, max_length_src=MAX_SEQUENCE_LENGTH_SRC, max_length_trg=MAX_SEQUENCE_LENGTH_IN_TRG):
    # source preprocessing
    # input_sequence = src_preprocessing(input_sequence)
    
    # Tokenize and pad the input sequence
    input_seq = tokenizer_src.texts_to_sequences([input_sequence])
    input_seq = pad_sequences(input_seq, maxlen=max_length_src, padding='post')
    
    # Initialize target sequence with the start token
    target_seq = np.zeros((1, max_length_trg))
    target_seq[0, 0] = tokenizer_trg.word_index['<START>']
    
    # Generate the translation
    output_sequence = []
    for i in range(max_length_trg - 1):
        output_tokens = model.predict([input_seq, target_seq])
        sampled_token_index = np.argmax(output_tokens[0, i, :])
        sampled_token = tokenizer_trg.index_word.get(sampled_token_index, '<UKN>')
        
        if sampled_token == '<END>':
            break
        
        output_sequence.append(sampled_token)
        
        # Update target sequence for next token prediction
        target_seq[0, i+1] = sampled_token_index
    
    return ' '.join(output_sequence)
    
    


