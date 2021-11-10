import params as pp

SEQ_LEN = 80
NEG_SEQ_LEN = 10
USER_PRODUCT_VECTOR_MATCH_LEN = 64
USER_QUIZ_VECTOR_LEN = 64
METADATA_TOKEN_LEN = 4000


class ModelParams(pp.Params):
    seq_len = SEQ_LEN
    neg_seq_len = NEG_SEQ_LEN
    user_product_vector_match_len = USER_PRODUCT_VECTOR_MATCH_LEN
    user_quiz_vector_len = USER_QUIZ_VECTOR_LEN
    uniq_metadata_tokens_len = METADATA_TOKEN_LEN
    encoder_num = 1
    head_num = 2
    embedding_dim = 64
    feed_forward_units = 64
    dropout_rate = 0.01
    max_days_before = 500
    flg_add_original_inputs = True
    mask_embedding_output = False
    batch_size = 64
    metadata_vocab_len = -1
    brand_vocab_len = -1
    category_vocab_len = -1
    user_product_match_vocab_len = -1
    program_types_vocab_len = -1
