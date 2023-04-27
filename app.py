import re
import pickle
import tensorflow as tf
import numpy as np

from enc_dec import Encoder, Decoder
from keras import Input
from keras.utils import pad_sequences
from flask import Flask, render_template, request, jsonify



with open('data1000.pkl', 'rb') as f:
    data = pickle.load(f)

batak_train = data['batak_train']
batak_test = data['batak_test']
indo_train = data['indo_train']
indo_test = data['indo_test']
tokenizer_indo = data['tokenizer_indo']
tokenizer_batak = data['tokenizer_batak']
batak_max = data['batak_max']
indo_max = data['indo_max']
batak_vocab = data['batak_vocab']
indo_vocab = data['indo_vocab']


def preprocessing(sentence):
    # datacleaning
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿' ']+", "", sentence)
    sentence = re.sub(r'\s{2,}', ' ', sentence)
    # casefolding
    sentence = sentence.strip().lower()
    # sos dan eos
    sentence = '<sos> ' + sentence + ' <eos>'

    return sentence


class MyModel(tf.keras.Model):
    def __init__(self, ind_vocab, btt_vocab, embedding_dim, units, **kwargs):
        super().__init__()  # oop python https://stackoverflow.com/a/27134600/4084039
        self.encoder = Encoder(ind_vocab, embedding_dim, units)
        self.decoder = Decoder(btt_vocab, embedding_dim, units)
        self.state_h = 0
        self.batch_loss = 0
        self.units = units

    def call(self, input_shape, training=False):
        encoder_input = input_shape
        print(f'input shape: {input_shape}')
        # hidden layer initiation (* = 0)
        state_h = tf.zeros((1, self.units))
        encoder_output, state_h = self.encoder(encoder_input, state_h)
        decoder_input = Input(shape=(1, ))
        attention_wts, decoder_output, state_h, _ = self.decoder(decoder_input, encoder_output, state_h)
        return decoder_output

    def translate(self, input_sentence=''):
        if input_sentence == '':
            # random number -> k
            k = np.random.randint(len(indo_train))
            # take a random sentence from the output language
            w = indo_train[k]
            # converts the input sequence to text and removes it from the list
            asked = tokenizer_indo.sequences_to_texts([w])[0]
            # displays the input text sequence without the <sos> and <eos> tags
            print(f"input sentence: {' '.join(asked.split(' ')[1:-1])}")

            # pick up a random sentence from the target language
            o = batak_train[k]
            # change the input sequence to text
            o = tokenizer_batak.sequences_to_texts([o])[0]
            # displays input text sequence
            print(f"actual translation: {' '.join(o.split(' ')[1:-1])}")

        else:
            inp = input_sentence
            preprocessed = preprocessing(inp)
            ind_tkn = tokenizer_indo.texts_to_sequences([preprocessed])
            w = pad_sequences(ind_tkn, maxlen=indo_max, padding='post')[0]
            asked = tokenizer_indo.sequences_to_texts([w])[0]  # mengubah input sequence menjadi text
            print(f"input sentence: {' '.join(asked.split(' ')[1:-1])}")  # menampilkan input text sequence

        # mempersiapkan data untuk model
        inputs = tf.convert_to_tensor(
            w.reshape(1, -1))  # inisialisasi input tensor reshape w(18, ) --> (1, 18) 1 dimensi sebelum dimensi ke 1
        state_h = tf.zeros((1, self.units))  # inisialisasi hidden layer (1, 1024)
        encoder_output, state_h = self.encoder(inputs, state_h)
        decoder_input = tf.expand_dims(tokenizer_indo.texts_to_sequences(['<sos>'])[0],
                                       0)  # inisialisasi input decoder <sos> dan mengeluarkannya dari list[0]
        # proses penerjemahan
        attention = []
        result = ''
        for i in range(batak_max):
            attention_wts, decoder_output, state_h, _ = self.decoder(decoder_input, encoder_output, state_h)
            attention_wts = attention_wts.numpy().flatten()  # list --> narrayobject 1 dimensi
            predicted_word = tf.argmax(decoder_output[0]).numpy()  # averagemax (1, 2354) --> (2354, )
            if predicted_word == tokenizer_indo.texts_to_sequences(['<eos>'])[0][
                0]:  # proses menerjemahkan selesai jika ada <eos>
                break
            result += tokenizer_batak.sequences_to_texts([[predicted_word]])[0] + ' '
            decoder_input = tf.expand_dims([predicted_word], 0)  # keluaran digunakan sebagai masukan lagi pada decoder
            attention.append(attention_wts[1:len(asked.split(' ')) - 1])
        print(f'predicted translation: {result}')

        return result
    

# Hyperparameters
units = 512
embedding_dim = 128
BATCH_SIZE = 20
BUFFER_SIZE = 100

# Initiating model
dataset = tf.data.Dataset.from_tensor_slices((indo_train, batak_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

model = MyModel(indo_vocab, batak_vocab, embedding_dim, units)
model.build(input_shape=(1, 18))

model.load_weights("weight1000.h5")



app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    input_sentence = str(request.args['s'])
    data = {
        "input": input_sentence,
        "output": model.translate(input_sentence),
    }
    return jsonify(data)

# @app.route('/translate', methods=['POST'])
# def translate():
#     input_sentence = request.form['input-sentence']
#     output_sentence = model.translate(input_sentence)
#     return output_sentence


if __name__ == "__main__":
    app.run(debug=True)