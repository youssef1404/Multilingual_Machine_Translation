from models.model_por import Transformer
import tensorflow as tf
import pickle

# Set hyperparamters for the model
D_MODEL = 512 # 512
N_LAYERS = 4 # 6
FFN_UNITS = 512 # 2048
N_HEADS = 8 # 8
DROPOUT_RATE = 0.1 # 0.1
MAX_LENGTH = 12
sos_token_input = 13945
eos_token_input = 13946
sos_token_output = 9238
eos_token_output = 9239


def load_tokenizer(filename):
    """Load the tokenizer from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load Tokenizers
tokenizer_inputs = load_tokenizer('../artifacts/por_model/tokenizer_inputs.pkl')
tokenizer_outputs = load_tokenizer('../artifacts/por_model/tokenizer_outputs.pkl')

# Create the Transformer model
transformer = Transformer(vocab_size_enc=13947,
                        vocab_size_dec=9240,
                        d_model=D_MODEL,
                        n_layers=N_LAYERS,
                        FFN_units=FFN_UNITS,
                        n_heads=N_HEADS,
                        dropout_rate=DROPOUT_RATE)

# Create a checkpoint to restore only the model
ckpt = tf.train.Checkpoint(transformer=transformer)

# Create a CheckpointManager to manage checkpoints
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Restore the latest checkpoint
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print("Model restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))
else:
    print("No checkpoint found. Please check the path.")


def predict(inp_sentence, tokenizer_in, tokenizer_out, target_max_len):
    # Tokenize the input sequence using the tokenizer_in
    inp_sentence = sos_token_input + tokenizer_in.encode(inp_sentence) + eos_token_input
    enc_input = tf.expand_dims(inp_sentence, axis=0)

    # Set the initial output sentence to sos
    out_sentence = sos_token_output
    # Reshape the output
    output = tf.expand_dims(out_sentence, axis=0)

    # For max target len tokens
    for _ in range(target_max_len):
        # Call the transformer and get the logits 
        predictions = transformer(enc_input, output, False) #(1, seq_length, VOCAB_SIZE_ES)
        # Extract the logits of the next word
        prediction = predictions[:, -1:, :]
        # The highest probability is taken
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        # Check if it is the eos token
        if predicted_id == eos_token_output:
            return tf.squeeze(output, axis=0)
        # Concat the predicted word to the output sequence
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def translate(sentence):
    # Get the predicted sequence for the input sentence
    output = predict(sentence, tokenizer_inputs, tokenizer_outputs, MAX_LENGTH).numpy()
    # Transform the sequence of tokens to a sentence
    predicted_sentence = tokenizer_outputs.decode(
        [i for i in output if i < sos_token_output]
    )

    return predicted_sentence
