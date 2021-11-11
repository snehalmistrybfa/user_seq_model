# Databricks notebook source
from keras import backend
from tensorflow.keras.layers import *
import tensorflow as tf


def get_product_model_layers(params):
    emb_dims = params.embedding_dim
    return {
        "metadata": tf.keras.Sequential([
            Embedding(params.metadata_vocab_len + 2,
                      emb_dims, mask_zero=True,
                      name="metadata_token_embeddings", input_shape=(None, None)),
            Lambda(lambda x: backend.mean(x, axis=[2])),
            # tf.keras.layers.LSTM(32)
        ]),
        "brand": tf.keras.Sequential([
            Embedding(params.brand_vocab_len + 2,
                      emb_dims, mask_zero=True, name="brand_embeddings")
        ]),
        "category": tf.keras.Sequential([
            Embedding(params.category_vocab_len + 1,
                      emb_dims, mask_zero=True, name="category_embeddings")
        ]),
        "price": tf.keras.Sequential([

        ]),
        "user_product_match_embs": tf.keras.Sequential([
            Embedding(params.program_types_vocab_len + 2, emb_dims, mask_zero=True,
                      name="user_product_match_embeddings")
        ]),
        "program_types": tf.keras.Sequential([
            Lambda(lambda x: tf.one_hot(x, depth=params.user_product_match_vocab_len + 2, name="program_one_hot"))
        ]),
        "compress_prod_feats": tf.keras.Sequential([
            Dense(emb_dims)
        ])
    }


class ProductFeaturesEncoder(Layer):
    def __init__(self, params):
        super(ProductFeaturesEncoder, self).__init__()
        self.product_model_layers = get_product_model_layers(params)

    def call(self, inputs, training=True):
        metadata_entry, brand_entry, category_entry, price_entry, user_product_match_entry, program_types_input = inputs
        return process_product_inputs(self.product_model_layers,
                                      metadata_entry,
                                      brand_entry,
                                      category_entry,
                                      price_entry,
                                      user_product_match_entry,
                                      program_types_input)

    def compute_mask(self, inputs, mask=None):
        # Index 0 is reserved for padding
        # Index 1 is reserved for UNKNOWN TOKEN
        # Index > 1 is used for valid tokens
        _, _, category_entry, _, _, _ = inputs
        return tf.not_equal(category_entry, 0)


def process_product_inputs(product_model_layers,
                           metadata_entry,
                           brand_entry,
                           category_entry,
                           price_entry,
                           user_product_match_entry,
                           program_types_input):
    sum_prod_embs = (product_model_layers["metadata"](metadata_entry) +
                     product_model_layers["brand"](brand_entry) +
                     product_model_layers["category"](category_entry))
    concat_prod_embs = tf.concat([sum_prod_embs,
                                  tf.expand_dims(price_entry, -1),
                                  product_model_layers["program_types"](program_types_input),
                                  product_model_layers["user_product_match_embs"](user_product_match_entry)

                                  ], axis=-1)

    return product_model_layers["compress_prod_feats"](concat_prod_embs)
