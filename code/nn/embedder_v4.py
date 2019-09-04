from utils.net_element import *
from utils.decorator import *
from dependency import *

"""
The embedder has three nested kernels, Key embedding, K, Weight embedding, W，and Value embedding, V.
K and W have the shape of [dims_g, dims_d, h, w, c], V has the shape of [dim_g, h, w, c]
"""

class EMBEDDER:
    def __init__(self, emb_shape, g_emb_size, d_emb_size, emb_norm, emb_type):
        self.emb_shape = emb_shape # [h, w, c]
        self.K_space, self.W_space, self.V_space = self.embeds_space([g_emb_size, d_emb_size]) # [g, d, h, w, c]
        self.emb_norm = emb_norm
        self.emb_type = emb_type
        # Weight and Value embedding
        self.K, self.W, self.V = self.embeds_inits() # [g, d, h, w, c]


    @lazy_method
    def embeds_space(self, emb_size):
        emb_shapes = emb_size + self.emb_shape
        w_shapes = [emb_size[0]] + self.emb_shape
        return emb_shapes, w_shapes, emb_shapes

    def embeds_inits(self):
        # K
        K = ne.weight_variable(self.K_space, "Key", init_type="TN")
        # V
        V = ne.weight_variable(self.V_space, "Value", init_type="HE") 
        # W
        W = ne.weight_variable(self.W_space, "Weight", init_type="TN")

        tf.summary.histogram("Key_embs", K)
        tf.summary.histogram("Weight_embs", W)
        tf.summary.histogram("Value_embs", V)
        return K, W, V


    @lazy_method
    def to_detail(self, inputs, k_embs, v_embs):
        def _to_dtl(embs): # embs: [2, d, h, w, c]
            k_embs, v_embs = tf.unstack(embs)
            _, d_coef = self.to_categorical(inputs, k_embs) # [batch, d, c]
            return self.to_embedded(v_embs, d_coef) # [batch, h, w, c]
        # inputs: [batch, h, w, c]; w_embs: [g, d, h, w, c], v_embs: [g, d, h, w, c]
        embs = tf.stack([k_embs, v_embs]) # [2, g, d, h, w, c]
        embs = tf.transpose(embs, [1, 0, 2, 3, 4, 5]) # [g, 2, d, h, w, c]
        detail = tf.map_fn(_to_dtl, embs) # [g, batch, h, w, c]
        detail = tf.transpose(detail, [1, 0, 2, 3, 4]) # [batch, g, h, w, c]
        return detail


    @lazy_method
    def to_categorical(self, inputs, embs, emb_type="SOFTMAX", channel_wise=True):
        if len(embs.get_shape().as_list()) == 4:
            e_shape = embs.get_shape().as_list() # [d, h, w, c] 
        else:
            e_shape = embs.get_shape().as_list()[1:] # [g, h, w, c]
        if channel_wise:
            shape = [-1, e_shape[0], np.prod(e_shape) // (e_shape[0]*e_shape[-1]), e_shape[-1]] # [batch, g, h*w, c]
        else:
            shape = [-1, e_shape[0], np.prod(e_shape) // e_shape[0], 1] # [batch, g, h*w*c, 1]
        
        inputs_ext = tf.expand_dims(inputs, 1)  #[batch, 1, h, w, c]
        diff = tf.norm(tf.reshape(inputs_ext - embs, shape), axis = -2) # [batch, g, h, w, c] --reshape--> [batch, g, h*w, c] --norm--> [batch, g, c]
        k_value = diff #[batch, g, c];  [batch, g, 1]
        if emb_type == "SOFTMAX":
            k_coef = tf.nn.softmax(-k_value, axis=1) # [batch, g, c];  [batch, g, 1]
        elif emb_type == "MINIMUM":
            idices = tf.argmin(k_value, axis=1) # [batch, 1, c];  [batch, 1, 1]
            #import pdb; pdb.set_trace()
            k_coef = tf.one_hot(idices, e_shape[0], on_value=1.0, off_value=0.0, axis=1) # [batch, g, c];  [batch, g, 1]
        return k_value, k_coef

    
    @lazy_method
    def to_embedded(self, embs, k_coef, length=4):
        if len(embs.get_shape().as_list()) == length:
            t_emb = tf.tile(tf.expand_dims(embs, 0), # [1, d, h, w, c]
                            [tf.shape(k_coef)[0]]+[1]*len(embs.get_shape().as_list())) # [batch, d, h, w, c]
            embs = t_emb

        k_coef_len = len(k_coef.get_shape().as_list())
        embs_len = len(embs.get_shape().as_list())
        for _ in range(embs_len - k_coef_len): # [batch, d, 1, 1, c]
            k_coef = tf.expand_dims(k_coef, -2) 
        #import pdb; pdb.set_trace()
        weighted = k_coef * embs # [batch, d, h, w, c]
        #import pdb; pdb.set_trace()
        embedded = tf.reduce_sum(weighted, axis=1) # [batch, h, w, c]
        return embedded

    
    @lazy_method
    def evaluate(self, inputs, is_training, g_coef=None, d_coef=None):
        net = inputs
        if g_coef == None and d_coef == None:
            if self.emb_norm == "BATCH":
                net = ne.batch_norm(net, is_training)
            elif self.emb_norm == "LAYER":
                net = ne.layer_norm(net, is_training)
            #import pdb; pdb.set_trace()
            self.detail = self.to_detail(net, self.K, self.V) # [batch, g, h, w, c]
            g_value, g_coef = self.to_categorical(net, self.W, self.emb_type, False)
            
            net = self.to_embedded(self.detail, g_coef) #[batch, h, w, c]
        elif g_coef != None and d_coef != None:
            if len(g_coef.get_shape().as_list()) != 3:
                g_coef = tf.expand_dims(g_coef, -1)
            g_weighted = self.to_embedded(self.V, g_coef, length=5) # [batch, d, h, w, c]

            if len(d_coef.get_shape().as_list()) != 3:
                d_coef = tf.expand_dims(d_coef, -1)
            net= self.to_embedded(g_weighted, d_coef) # [batch, h, w, c]

        else:
            raise NotImplemented

        if self.emb_norm == "BATCH":
            net = ne.batch_norm(net, is_training)
        elif self.emb_norm == "LAYER":
            net = ne.layer_norm(net, is_training)
        
        self.embedded = net
        return net, g_coef