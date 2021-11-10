from base_model import *
from tensorflow.keras.layers import *
import tensorflow as tf

from models.product_model import ProductFeaturesEncoder


def create_user_product_model_inputs(params):
    int_fixed_len_label_cols = ["label_user_product_match_str",
                                "label_program_types"]
    label_inputs_dict = dict(([get_fixed_len_input_tuple(c, params.neg_seq_len + 1) for c in int_fixed_len_label_cols] +
                              [("label_user_product_match", Input(name="user_product_match_labels",
                                                                  shape=(params.neg_seq_len + 1,
                                                                         params.user_product_vector_match_len),
                                                                  dtype=tf.int64))] +
                              get_product_feats_inputs("label_products", params.neg_seq_len + 1, params)))
    label_inputs_dict.update(create_model_base_inputs(params))
    return label_inputs_dict


def build_user_product_model(params):
    inputs = create_user_product_model_inputs(params)
    prod_input_label = [inputs["label_products_metadata"],
                        inputs["label_products_brand"],
                        inputs["label_products_category"],
                        inputs["label_products_price"],
                        inputs["label_user_product_match_str"],
                        inputs["label_program_types"]]

    user_embs, features_encoder = build_base_seq_model(params, inputs)

    # final_layer = Dense(params.max_prod_id_index)
    #   compress_quiz_feats = Dense(params.embedding_dim,
    #                               activation=layers.LeakyReLU())
    compress_user_embs = Dense(params.embedding_dim,
                               activation=LeakyReLU())

    extract_final_emb_layer = Lambda(lambda x: x[:, -1, :])
    user_emb = extract_final_emb_layer(user_embs)
    #   user_emb = tf.keras.layers.concatenate([user_emb,
    #                                           Dropout(params.dropout_rate)(compress_quiz_feats(inputs["user_quiz_features"]))])
    user_emb = Dropout(params.dropout_rate)(compress_user_embs(user_emb))
    el_mult_layer = Lambda(lambda x: tf.math.reduce_sum(tf.expand_dims(x[0], 1) * x[1], axis=-1))
    prod_labl_embs = features_encoder(prod_input_label)
    y = el_mult_layer([user_emb, prod_labl_embs])
    model = Model(inputs=inputs, outputs=y)
    y_pred = y
    return model, user_emb, y_pred
