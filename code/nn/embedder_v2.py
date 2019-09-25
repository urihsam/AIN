from utils.net_element import *
from utils.decorator import *
from dependency import *


class EMBEDDER:
    def __init__(self, emb_shape, g_emb_size, d_emb_size, emb_norm, emb_type):
        self.emb_shape = emb_shape # [h, w, c]
        self.emb_space, self.w_space = self.embeds_weights_space([g_emb_size, d_emb_size]) # [g, d, h, w, c], [g, d, 1, 1, c]
        self.emb_norm = emb_norm
        self.emb_type = emb_type
        # Weights
        self.embeds, self.weights = self.embeds_weights("TNORMAL", "W_emb_")
        self.weighted_embeds


    @lazy_property
    def weighted_embeds(self):
        return tf.reduce_mean(self.weights * self.embeds, axis=1) #[g, d, h, w, c] --> # [g, h, w, c]

    @lazy_method
    def embeds_weights_space(self, emb_size):
        emb_shapes = emb_size + self.emb_shape
        w_shapes = emb_size + [1]*len(self.emb_shape)
        return emb_shapes, w_shapes

    def embeds_weights(self, init_type, W_name):
        layer_idx = 0
        if init_type == "TNORMAL":
            emb_init = tf.truncated_normal_initializer(stddev=1.0)
            w_init = tf.truncated_normal_initializer(stddev=1.0)
        # emb
        _embeds = tf.get_variable(initializer=emb_init, shape=self.emb_space, name="Embedding")
        # weight
        W_key = "{}{}".format(W_name, layer_idx)
        _weights = tf.get_variable(initializer=w_init, shape=self.w_space, name=W_key)
        
        tf.summary.histogram("Embeds", _embeds)
        tf.summary.histogram("Weight_"+W_key, _weights)
        return _embeds, _weights


    @lazy_method
    def to_categorical(self, inputs, weighted_embs, emb_type="SOFTMAX"):
        we_shape = weighted_embs.get_shape().as_list() # [g, h, w, c]
        shape = [-1, we_shape[0], np.prod(we_shape) // we_shape[0]] # [batch, g, h*w*c]
        inputs_ext = tf.expand_dims(inputs, 1)  #[batch, 1, h, w, c]
        diff = tf.norm(tf.reshape(inputs_ext - weighted_embs, shape), axis = -1) # [batch, g, h, w, c] --reshape--> [batch, g, h*w*c] --norm--> [batch, g]
        k_value = diff #[batch, g]
        if emb_type == "SOFTMAX":
            k_coef = tf.nn.softmax(-k_value) # [batch, g]
        elif emb_type == "MINIMUM":
            idices = tf.argmin(k_value, axis=1) # [batch, 1]
            #import pdb; pdb.set_trace()
            k_coef = tf.one_hot(idices, we_shape[0], on_value=1.0, off_value=0.0, axis=-1) # [batch, g]
        return k_value, k_coef

    
    @lazy_method
    def to_embedded(self, weighted_embs, k_coef):
        w_emb = tf.tile(tf.expand_dims(weighted_embs, 0), # [1, g, h, w, c]
                        [tf.shape(k_coef)[0]]+[1]*len(weighted_embs.get_shape().as_list())) # [batch, g, h, w, c]
        for _ in range(len(self.emb_shape)): # [batch, g, 1, 1, 1]
            k_coef = tf.expand_dims(k_coef, -1) 
        weighted = k_coef * w_emb # [batch, g, h, w, c]
        #import pdb; pdb.set_trace()
        embedded = tf.reduce_sum(weighted, axis=1) # [batch, h, w, c]
        return embedded

    
    @lazy_method
    def evaluate(self, inputs, is_training, k_coef=None):
        net = inputs
        if k_coef == None:
            if self.emb_norm == "BATCH":
                net = ne.batch_norm(net, is_training)
            elif self.emb_norm == "LAYER":
                net = ne.layer_norm(net, is_training)

            k_value, k_coef = self.to_categorical(net, self.weighted_embeds, self.emb_type)
        
        net = self.to_embedded(self.weighted_embeds, k_coef) #[batch, h, w, c

        if self.emb_norm == "BATCH":
            net = ne.batch_norm(net, is_training)
        elif self.emb_norm == "LAYER":
            net = ne.layer_norm(net, is_training)
            
        self.embedded = net
        return self.embedded, k_coef
