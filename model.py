from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
import json
import io
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape(
                                       (input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape(
                                       (input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape(
                                       (input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        # Be sure to call this at the end
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(
                states, type(states))
            assert isinstance(states, list) or isinstance(
                states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            batch_size = encoder_out_seq.shape[0]
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(
                K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>', W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(
                K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>', U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(
                K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh,
                            self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.zeros_like(inputs)
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            # <= (batch_size, latent_dim
            fake_state = K.tile(fake_state, [1, hidden_size])
            return fake_state

        fake_state_c = create_inital_state(
            encoder_out_seq, encoder_out_seq.shape[-1])
        # <= (batch_size, enc_seq_len, latent_dim
        fake_state_e = create_inital_state(
            encoder_out_seq, encoder_out_seq.shape[1])

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape(
                (input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape(
                (input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]


def load_saved_model(dir_hash):
    with open('/h5.models/'+dir_hash+'/model_params.json', 'r') as f:
        for line in f:
            data = json.loads(line)
    with open('/h5.models/'+dir_hash+'/source_tokenizer.json', encoding='utf-8') as f:
        temp = json.load(f)
        source_tokenizer = tokenizer_from_json(temp)
    with open('/h5.models/'+dir_hash+'/target_tokenizer.json', encoding='utf-8') as f:
        temp = json.load(f)
        target_tokenizer = tokenizer_from_json(temp)
    full_model = load_model('/h5.models/'+dir_hash+'/full_model.h5',
                            custom_objects={'AttentionLayer': AttentionLayer})
    encoder_model = load_model('/h5.models/'+dir_hash+'/encoder_model.h5',
                               custom_objects={'AttentionLayer': AttentionLayer})
    decoder_model = load_model('/h5.models/'+dir_hash+'/decoder_model.h5',
                               custom_objects={'AttentionLayer': AttentionLayer})
    return data, source_tokenizer, target_tokenizer, full_model, encoder_model, decoder_model


def translate(sentence, encoder_model, decoder_model, source_tokenizer, target_tokenizer, src_vsize, tgt_vsize, source_timesteps, target_timesteps):
    target = "sentencestart"
    source_text_encoded = source_tokenizer.texts_to_sequences([sentence])
    target_text_encoded = target_tokenizer.texts_to_sequences([target])
    source_preproc_text = pad_sequences(
        source_text_encoded, padding='post', maxlen=source_timesteps)
    target_preproc_text = pad_sequences(
        target_text_encoded, padding='post', maxlen=1)
    encoder_out, enc_last_state1, enc_last_state2 = encoder_model.predict(
        source_preproc_text)
    continuePrediction = True
    output_sentence = ''
    total = 0
    while continuePrediction:
        decoder_pred, attn_state, decoder_state1, decoder_state2 = decoder_model.predict(
            [encoder_out, target_preproc_text, enc_last_state1, enc_last_state2])
        index_value = np.argmax(decoder_pred, axis=-1)[0, 0]
        sTemp = target_tokenizer.index_word.get(index_value, 'UNK')
        output_sentence += sTemp + ' '
        total += 1
        if total >= target_timesteps or sTemp == 'sentenceend':
            continuePrediction = False
        enc_last_state1 = decoder_state1
        enc_last_state2 = decoder_state2
        target_preproc_text[0, 0] = index_value
    return output_sentence


# choice = input(
#     "Enter nstd-std for Non-Standard to Standard Translation or std-nstd Standard to Non-Standard Translation: ")

dir_hash = "nstd-std"

model_dict, source_tokenizer, target_tokenizer, _, encoder_model, decoder_model = load_saved_model(
    dir_hash)

src_vsize = model_dict['SourceVocab']
tgt_vsize = model_dict['TargetVocab']
SOURCE_TIMESTEPS = model_dict['SourceTimeSteps']
TARGET_TIMESTEPS = model_dict['TargetTimeSteps']
HIDDEN_SIZE = model_dict['HiddenSize']
EMBEDDING_DIM = model_dict['EmbeddingDim']


def lr(text):
    sentence = text
    translation = translate(sentence, encoder_model, decoder_model, source_tokenizer, target_tokenizer, src_vsize,
                            tgt_vsize, SOURCE_TIMESTEPS, TARGET_TIMESTEPS)
    translation.replace(" sentenceend", "")
    return translation.replace(" sentenceend", "")


def rl(text):
    sentence = text
    translation = translate(sentence, encoder_model, decoder_model, source_tokenizer, target_tokenizer, src_vsize,
                            tgt_vsize, SOURCE_TIMESTEPS, TARGET_TIMESTEPS)
    return translation.replace(" sentenceend", "")


# sentence = input("Please enter a non standard Filipino text: ")
# rl(sentence)
