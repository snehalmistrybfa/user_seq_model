from collections import defaultdict
import tensorflow as tf
from datetime import datetime, timedelta

from keras.layers import TextVectorization, StringLookup, Normalization
from tensorflow.python.data.experimental.ops import readers
import numpy as np
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras.layers import *
from collections import OrderedDict


def prepare_product_lookup_hashtables(product_metata_dicts,
                                      uniq_user_product_match_str,
                                      uniq_program_types,
                                      uniq_metadata_tokens_len):
    product_meta_indices_dicts, pre_process_steps = convert_to_indices(product_metata_dicts,
                                                                       uniq_user_product_match_str,
                                                                       uniq_program_types,
                                                                       uniq_metadata_tokens_len)
    product_metadata_lookup_dict = OrderedDict(
        [(k, get_hash_table(k, info)) for k, info in product_meta_indices_dicts.items()])
    return product_metadata_lookup_dict, pre_process_steps


def convert_to_indices(product_metata_dicts,
                       uniq_user_product_match_str,
                       uniq_program_types,
                       uniq_metadata_tokens_len):
    preprocess_price_norm = Normalization(name="preprocess_price_norm", axis=None)
    preprocess_price_norm.adapt(tf.constant(list(product_metata_dicts["price"].values())))

    pre_process_steps = OrderedDict({"metadata": init_text_vectorizer("preprocess_prod_md_vectorizer",
                                                                      list(product_metata_dicts["metadata"].values()),
                                                                      uniq_metadata_tokens_len),
                                     "brand": init_stringlookup("preprocess_brand_str_lookup",
                                                                list(product_metata_dicts["brand"].values())),
                                     "category": init_stringlookup("preprocess_category_str_lookup",
                                                                   list(product_metata_dicts["category"].values())),
                                     "price": preprocess_price_norm,
                                     "pid": init_stringlookup("product_id_to_int",
                                                              list(product_metata_dicts["brand"].keys())),
                                     "user_product_match": init_stringlookup("user_product_match_lookup",
                                                                             uniq_user_product_match_str),
                                     "program_type": init_stringlookup("program_types_lookup", uniq_program_types)})
    product_meta_indices_dicts = OrderedDict(
        [(k, dict(zip(d.keys(), pre_process_steps[k](tf.constant(list(d.values())))))) for k, d in
         product_metata_dicts.items()])
    return product_meta_indices_dicts, pre_process_steps


def init_stringlookup(name, vals):
    string_lookup_obj = StringLookup(name=name, mask_token="PAD")
    string_lookup_obj.adapt(tf.constant(vals))
    return string_lookup_obj


def init_text_vectorizer(name, vals, uniq_metadata_tokens_len):
    obj = TextVectorization(max_tokens=uniq_metadata_tokens_len, name=name)
    obj.adapt(tf.constant(vals))
    return obj


class HashTableWithArray(tf.keras.layers.Layer):
    # Make sure PAD is first entry for keys
    def __init__(self, keys, vals, name="product_id_table"):
        self.key_to_index = init_stringlookup(name + "_index_lookup", keys)
        self.output_dim = len(vals[0])
        vals_aligned_np = np.zeros((len(keys) + 1, self.output_dim), dtype=np.int64)
        for k, index, v in zip(keys, self.key_to_index(keys).numpy(), vals):
            vals_aligned_np[index] = v
        self.index_to_vals = Embedding(input_dim=len(keys) + 1,
                                       output_dim=self.output_dim,
                                       embeddings_initializer=tf.keras.initializers.Constant(vals_aligned_np),
                                       trainable=False,
                                       mask_zero=True,
                                       dtype=tf.int64)

    def lookup(self, inputs):
        return self.index_to_vals(self.key_to_index(inputs))

    def call(self, inputs):
        return self.lookup(inputs)


def get_hash_table(k, info):
    def_val = 0 if k == "price" else 1
    if k == "metadata":
        return HashTableWithArray(list(info.keys()),
                                  list(info.values()),
                                  name="metadata")
    else:
        return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(list(info.keys()),
                                                                             list(info.values())),
                                         default_value=def_val)
