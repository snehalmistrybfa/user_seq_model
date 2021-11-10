from keras import backend
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
from tensorflow import Tensor
from tensorflow.keras.models import Model

from models.model_params import ModelParams
from models.product_model import ProductFeaturesEncoder

product_fixed_len_feats_cols = ["brand", "category", "price"]


def create_model_base_inputs(params):
    int_fixed_len_cols = ["hist_program_types",
                          "hist_user_product_match_str",
                          "hist_days_before",
                          "hist_days_diff"]
    all_inputs = dict([get_fixed_len_input_tuple(c, params.seq_len) for c in int_fixed_len_cols] + \
                      [("hist_user_product_match", Input(name="hist_user_product_match",
                                                         shape=(params.seq_len, params.user_product_vector_match_len),
                                                         dtype=tf.int64)),
                       ("user_quiz_features", Input(name="user_quiz_features",
                                                    shape=(params.user_quiz_vector_len), dtype=tf.int64))] +
                      get_product_feats_inputs("hist_products", params.seq_len, params))
    return all_inputs

def build_base_seq_model(params, inputs):
    features_encoder = ProductFeaturesEncoder(params)
    prod_input = [inputs["hist_products_metadata"],
                  inputs["hist_products_brand"],
                  inputs["hist_products_category"],
                  inputs["hist_products_price"],
                  inputs["hist_user_product_match_str"],
                  inputs["hist_program_types"]]
    prod_embs = features_encoder(prod_input)
    days_before_encoding = positional_encoding(params.max_days_before,
                                               params.embedding_dim)
    days_before_embs = tf.gather_nd(days_before_encoding[0, :, :],
                                    tf.expand_dims(inputs[f'hist_days_before'], axis=-1))
    user_embs = prod_embs + days_before_embs
    # user_embs = prod_embs
    prod_embs_mask = features_encoder.compute_mask(prod_input)
    # user_embs = Dropout(params.dropout_rate)(prod_embs)

    for _ in range(params.encoder_num):
        user_embs = encoder_block(user_embs,
                                  tf.expand_dims(prod_embs_mask, -1),
                                  params)
    return user_embs, features_encoder

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def encoder_block(attention_input,
                  attention_mask,
                  params: ModelParams) -> Tensor:
    attention_x = MultiHeadAttention(
        params.head_num,
        params.embedding_dim,
        use_bias=True
    )(attention_input,
      attention_input,
      attention_mask=attention_mask)
    attention_x = Dropout(params.dropout_rate)(attention_x)
    attention_x = Add()([attention_input, attention_x])
    # feed_forward_input = LayerNormalization(trainable=True)(attention_x)
    feed_forward_input = LayerNormalization()(attention_x)

    # Fully-connected layers.
    for num_units in [params.embedding_dim]:
        feed_forward_x = Dense(num_units)(feed_forward_input)
        feed_forward_x = BatchNormalization()(feed_forward_x)
        feed_forward_x = LeakyReLU()(feed_forward_x)
        feed_forward_x = Dropout(params.dropout_rate)(feed_forward_x)

    if params.flg_add_original_inputs:
        feed_forward_x = Add()([feed_forward_input, feed_forward_x])
    block_output = LayerNormalization()(feed_forward_x)
    return block_output


def get_fixed_len_input_tuple(c, seq_len, dtype=tf.int64):
    return (c, Input(name=c, shape=(seq_len), dtype=dtype))


def get_product_feats_inputs(col_prefix, seq_len, params):
    def get_dtype(c):
        return tf.float32 if c == "price" else tf.int64

    return [get_fixed_len_input_tuple(f"{col_prefix}_{c}", seq_len, get_dtype(c)) for c in
            product_fixed_len_feats_cols] + \
           [(f"{col_prefix}_metadata", Input(name=f"{col_prefix}_metadata",
                                             shape=(seq_len, params.uniq_metadata_tokens_len),
                                             dtype=tf.int64))]