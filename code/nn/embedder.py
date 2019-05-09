from utils.decorator import *
from dependency import *


class EMBEDDER:
    def __init__(self, emb_shape, num_category, emb_norm):
        self.emb_shape = emb_shape #[h, w, c]
        self.num_category = num_category #[K1, K2,..]
        self.emb_norm = emb_norm

        self.embeds_space
        self.embeds = self.embeds_weights()


    @lazy_property
    def embeds_space(self):
        shapes = []
        for k in self.num_category:
            shapes.append([k] + self.emb_shape)
        return shapes

    def embeds_weights(self, W_name="W_emb_"):
        n_layer = len(self.num_category)
        _weights = {}
        initializer = tf.truncated_normal_initializer(stddev=1.0)
        for layer_idx in range(n_layer):
            shape = self.embeds_space[layer_idx]
            W_key = "{}{}".format(W_name, layer_idx)
            _weights[W_key] = tf.get_variable(initializer=initializer, shape=shape, name=W_key)

            tf.summary.histogram("Weight_"+W_key, _weights[W_key])
        return _weights


    @lazy_method
    def to_categorical(self, inputs, embeds, emb_space):
        """inputs_ext = tf.expand_dims(inputs, -1) #[batch, h, w, 1, D]
        diff = tf.norm(inputs_ext - self.embeds, axis = -1) #[batch, h, w, K, D] --norm--> [batch, h, w, K]
        k = tf.argmin(diff, axis = -1) #[batch, h, w, 1] #[batch, 1]"""
        shape = [-1, emb_space[0], np.prod(emb_space) // emb_space[0]] # [batch, K, h*w*c]
        inputs_ext = tf.expand_dims(inputs, 1)  #[batch, 1, h, w, c]
        diff = tf.norm(tf.reshape(inputs_ext - embeds, shape), axis = -1) # [batch, K, h, w, c] --reshape--> [batch, K, h*w*c] --norm--> [batch, K]
        k_value = diff #[batch, K]
        k_coef = tf.nn.softmax(-k_value) # [batch, K]
        return k_value, k_coef

    
    @lazy_method
    def to_embedded(self, embeds, k_coef):
        embedded = tf.tile(tf.expand_dims(embeds, 0), [tf.shape(k_coef)[0]]+[1]*len(embeds.get_shape().as_list())) # [batch, K, h, w, c]
        for _ in range(len(self.emb_shape)): # [batch, k, 1, 1, 1]
            k_coef = tf.expand_dims(k_coef, -1) 
        weighted = k_coef * embedded
        #import pdb; pdb.set_trace()
        top_emb = tf.reduce_mean(weighted, axis=1)
        return top_emb

    @lazy_method
    def emb_layer(self, inputs, layer_idx, W_name="W_emb_"):
        W_key = "{}{}".format(W_name, layer_idx)
        embeds = self.embeds[W_key]
        emb_space = self.embeds_space[layer_idx]
        k_value, k_coef = self.to_categorical(inputs, embeds, emb_space)
        embedded = self.to_embedded(embeds, k_coef) #[batch, h, w, c]
        return embedded, k_value, k_coef

    
    @lazy_method
    def evaluate(self, inputs, is_training, res_start=0, layer_start=0):
        n_layer = len(self.num_category)
        net = inputs
        self.k_values = []; self.k_coefs = []
        

        for layer_idx in range(layer_start, n_layer):
            if self.emb_norm == "BATCH":
                net = ne.batch_norm(net, is_training)
            elif self.emb_norm == "LAYER":
                net = ne.layer_norm(net, is_training)
            res_net = net
            net, k_value, k_coef = self.emb_layer(net, layer_idx)
            self.k_values.append(k_value)
            self.k_coefs.append(k_coef)
            if layer_idx >= res_start:
                net += res_net
        if self.emb_norm == "BATCH":
            net = ne.batch_norm(net, is_training)
        elif self.emb_norm == "LAYER":
            net = ne.layer_norm(net, is_training)
            
        self.embedded = net
        return self.embedded, self.k_values, self.k_coefs
