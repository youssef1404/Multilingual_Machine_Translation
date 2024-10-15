import unicodedata
import re, os, joblib, dill
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import EncoderRNN, AttnDecoderRNN

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 12
hidden_size = 256

def load_checkpoint(checkpoint_path, encoder, decoder, device=device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)

    # Move models to the specified device
    encoder.to(device)
    decoder.to(device)

    print(f"Loaded checkpoint from {checkpoint_path}")

# When loading your Lang objects:
def load_objects(obj1, obj2):
    with open(os.path.join(os.getcwd(), 'artifacts', f'{obj1}.dill'), 'rb') as f:
        object1 = dill.load(f)
    with open(os.path.join(os.getcwd(), 'artifacts', f'{obj2}.dill'), 'rb') as f:
        object2 = dill.load(f)
    return object1, object2

# Load objects
input_lang, output_lang = load_objects('input_lang', 'output_lang')
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
load_checkpoint(os.path.join(os.getcwd(), 'artifacts', 'checkpoint_epoch_100.pt'), encoder, decoder)


def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, 0) for word in sentence.strip().split() if word]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def translate(sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])

        output_sentence = ' '.join(decoded_words)
    return output_sentence



