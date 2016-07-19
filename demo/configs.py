import tensorflow as tf

def tf_flag():
    flags = tf.app.flags
    flags.DEFINE_integer("idim", 300, "sentence embedding dimension")
    flags.DEFINE_integer("edim", 300, "encoding dimension")
    flags.DEFINE_integer("nhop", 3, "number of hops")
    flags.DEFINE_integer("nanswer", 5, "number of answer sentences")
    flags.DEFINE_integer("nstory", 60, "number of story sentences")
    flags.DEFINE_float("init_lr", 5e-3, "initial learning rate")
    flags.DEFINE_float("init_std", 0.05, "weight initialization std")
    flags.DEFINE_float("gamma", 0 , 'weight decay gamma value')
    flags.DEFINE_integer("batch_size", 32, 'Training batch size')
    return flags

