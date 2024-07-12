from seahorse.data.data_prep.llava_pretrain_cc3m import make_llava_pretrain_cc3m
from seahorse.data.data_prep.the_cauldron import make_the_cauldron

DATASET_REGISTRY = {
    "the_cauldron": make_the_cauldron,
    "llava_pretrain_cc3m": make_llava_pretrain_cc3m,
}
