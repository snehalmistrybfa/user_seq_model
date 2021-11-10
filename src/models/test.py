from collections import defaultdict
import tensorflow as tf
from datetime import datetime, timedelta
from tensorflow.python.data.experimental.ops import readers
import numpy as np
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras.layers import *
import random
from collections import OrderedDict
from models.preprocess import prepare_product_lookup_hashtables
from models.model_params import ModelParams

### 1.1 Get Product Metadata (Dummy)
random.seed(10)

num_products = 7000
# Sample product_metata_dicts
product_metata_dicts = OrderedDict({"metadata":
                                        OrderedDict([("PAD", "")] + [("p" + str(i), " ".join(
                                            ["md" + str(random.randint(0, 40)) for x in range(4)])) for i in
                                                                     range(num_products)]),
                                    "brand": OrderedDict(
                                        [("PAD", "PAD")] + [("p" + str(i), "brand" + str(random.randint(0, 20))) for i
                                                            in range(num_products)]),
                                    "category": OrderedDict(
                                        [("PAD", "PAD")] + [("p" + str(i), "cat" + str(random.randint(0, 20))) for i in
                                                            range(num_products)]),
                                    "price": OrderedDict([("PAD", 0)] + [("p" + str(i), random.randint(3, 11)) for i in
                                                                         range(num_products)])})

uniq_user_product_match_str = ["[0,0,0,1,0]",
                               "[0,0,0,0,1]",
                               "[1,0,0,1,0]"]
uniq_program_types = ["pa", "pb"]
uniq_metadata_tokens_len = 4000

product_metadata_lookup_dict, pre_process_steps = prepare_product_lookup_hashtables(product_metata_dicts,
                                                                                    uniq_user_product_match_str,
                                                                                    uniq_program_types,
                                                                                    uniq_metadata_tokens_len)

# Verify PADDING, UNKNOWN tokens
for k, action in product_metadata_lookup_dict.items():
    print(action.lookup(tf.constant([["p100", "p101", "p6", "hey"],
                                     ["PAD", "p1", "p2", "p62"]])))

params = ModelParams(seq_len=4,
                     neg_seq_len=3,
                     user_product_vector_match_len=5,
                     user_quiz_vector_len=7,
                     uniq_metadata_tokens_len=4,
                     metadata_vocab_len=len(pre_process_steps["metadata"].get_vocabulary()),
                     brand_vocab_len=len(pre_process_steps["brand"].get_vocabulary()),
                     category_vocab_len=len(pre_process_steps["category"].get_vocabulary()),
                     user_product_match_vocab_len=len(pre_process_steps["user_product_match"].get_vocabulary()),
                     program_types_vocab_len=len(pre_process_steps["program_type"].get_vocabulary()),
                     encoder_num=1,
                     head_num=1,
                     embedding_dim=64,
                     feed_forward_units=64,
                     dropout_rate=0.01,
                     max_days_before=500,
                     batch_size=2
                     )

## 2. Combine Product Inputs to get product embeddings
# Dummy Row
from copy import deepcopy

NEG_SEQ_LEN = 3
# User HashMap for item_types and program_types
# What do we do for padding...
samp_row = {"hist_products": tf.convert_to_tensor(["PAD", "p1", "p2", "p3"]),
            "hist_item_types": tf.convert_to_tensor(["PAD", "aa", "bb", "aa"]),
            "hist_program_types": tf.convert_to_tensor(["PAD", "pa", "pa", "pa"]),
            "hist_user_product_match": tf.convert_to_tensor([[0, 0, 0, 0, 0],
                                                             [0, 0, 0, 1, 0],
                                                             [0, 0, 0, 0, 1],
                                                             [1, 0, 0, 1, 0]]),
            "hist_user_product_match_str": tf.convert_to_tensor([
                "PAD",
                "[0,0,0,1,0]",
                "[0,0,0,0,1]",
                "[1,0,0,1,0]"]),
            "hist_days_before": tf.convert_to_tensor([0, 1, 2, 3]),
            "hist_days_diff": tf.convert_to_tensor([0, 1, 2, 3]),
            "label_product": tf.convert_to_tensor("p9"),
            "label_products": tf.convert_to_tensor(["p9"] + ["p" + str(i + 11) for i in range(NEG_SEQ_LEN)]),
            "label_user_product_match": tf.convert_to_tensor([[0, 0, 0, 1, 0],
                                                              [0, 0, 0, 0, 1],
                                                              [1, 0, 0, 1, 0],
                                                              [1, 0, 0, 1, 0]]),
            "label_user_product_match_str": tf.convert_to_tensor(["[0,0,0,1,0]",
                                                                  "[0,0,0,0,1]",
                                                                  "[1,0,0,1,0]",
                                                                  "[1,0,0,1,0]"]),
            "label_program_types": tf.convert_to_tensor("pb"),
            "user_quiz_features": tf.convert_to_tensor([0, 1, 0, 1, 0, 1, 0]),
            }


# Dummy Dataset to test mapping function
def generate_products_metadata(row, col_prefix, prod_seqs):
    for k, v in product_metadata_lookup_dict.items():
        row[f"{col_prefix}_{k}"] = v.lookup(prod_seqs)
    row[f"{col_prefix}_price"] = pre_process_steps["price"](row[f"{col_prefix}_price"])


def populate_product_metadata(row):
    generate_products_metadata(row, "hist_products", row["hist_products"])
    generate_products_metadata(row, "label_products", row["label_products"])
    row["label_program_types"] = pre_process_steps["program_type"](
        tf.repeat(tf.expand_dims(row["label_program_types"], -1), NEG_SEQ_LEN + 1, axis=-1))
    row["hist_user_product_match_str"] = pre_process_steps["user_product_match"](row["hist_user_product_match_str"])
    row["hist_program_types"] = pre_process_steps["program_type"](row["hist_program_types"])
    row["label_user_product_match_str"] = pre_process_steps["user_product_match"](row["label_user_product_match_str"])

    del row["hist_products"]
    del row["hist_item_types"]
    del row["label_product"]
    del row["label_products"]
    return row


samp_row2 = deepcopy(samp_row)
samp_row2["label_program_types"] = tf.convert_to_tensor("pa")
sampledataset = tf.data.Dataset.from_tensor_slices(
    pd.DataFrame.from_dict([samp_row, samp_row2, samp_row, samp_row]).to_dict(orient="list")).map(
    populate_product_metadata).batch(params.batch_size)

row = next(iter(sampledataset))
print(row)

# Model Inputs
from user_product_model import create_user_product_model_inputs

inputs = create_user_product_model_inputs(params)
print(inputs)

# Test Product Model
from product_model import ProductFeaturesEncoder

pfeats_encoder = ProductFeaturesEncoder(params)
output = pfeats_encoder([row["hist_products_metadata"],
                         row["hist_products_brand"],
                         row["hist_products_category"],
                         row["hist_products_price"],
                         row["hist_user_product_match_str"],
                         row["hist_program_types"]])
print(output._keras_mask)
print(output)

#Model Putting together
from user_product_model import build_user_product_model
model, user_emb, y_pred = build_user_product_model(params)

print(model(row))
#print(model.summary())