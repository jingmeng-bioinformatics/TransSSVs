import tensorflow.compat.v1 as tf
import numpy as np
import os
import glob
import argparse
from sklearn.utils import class_weight
import time
import pandas
from sklearn.metrics import matthews_corrcoef,f1_score
from sklearn.metrics import confusion_matrix


class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.5):
    super(Encoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers    
    self.pos_encoding = positional_encoding(1000, self.d_model)      
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training=True, mask=None):
    seq_len = tf.shape(x)[1]
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)
    enc_self_attns=[]
    for i in range(self.num_layers):
      x,attn_weights = self.enc_layers[i](x, training, mask)
      enc_self_attns.append(attn_weights)
    return x 

class Encoder_atten(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.5):
    super(Encoder_atten, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers    
    self.pos_encoding = positional_encoding(1000, self.d_model)      
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training=True, mask=None):
    seq_len = tf.shape(x)[1]
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)
    enc_self_attns=[]
    for i in range(self.num_layers):
      x,attn_weights = self.enc_layers[i](x, training, mask)
      enc_self_attns.append(attn_weights)
    return x ,enc_self_attns

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],np.arange(d_model)[np.newaxis, :],d_model)
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
  return tf.cast(pos_encoding, dtype=tf.float32)

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.5):
    super(EncoderLayer, self).__init__()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6) 
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training=True, mask=None):
    attn_output, attn_weights = self.mha(x, x, x, mask)  
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  
    ffn_output = self.ffn(out1) 
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  
    return out2,attn_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  
      tf.keras.layers.Dense(d_model)  
  ])

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask=None):
    batch_size = tf.shape(q)[0]
    q = self.wq(q)  
    k = self.wk(k)  
    v = self.wv(v) 
    
    q = self.split_heads(q, batch_size) 
    k = self.split_heads(k, batch_size)  
    v = self.split_heads(v, batch_size)  
    
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  
    output = self.dense(concat_attention)  
    return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask=None):
  matmul_qk = tf.matmul(q, k, transpose_b=True)  
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  
  output = tf.matmul(attention_weights, v)  
  return output, attention_weights


def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        p_t = y_true*y_pred + (tf.ones_like(y_true)-y_true)*(tf.ones_like(y_true)-y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true)-p_t),gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return binary_focal_loss_fixed

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_TransSSVs_model(shape_0,shape_1):
    inputESM=tf.keras.layers.Input(shape=(shape_0,shape_1)) #shape=(15,52)
    sequence=tf.keras.layers.Dense(512)(inputESM)
    sequence=tf.keras.layers.Dense(256)(sequence)
    sequence = Encoder(2, 256, 4, 64, rate=0.01)(sequence)
    shape_0_0=int((int(shape_0)-1)/2)
    sequence=sequence[:,shape_0_0,:]
    feature=tf.keras.layers.Dense(512,activation='relu')(sequence)
    feature=tf.keras.layers.Dense(256,activation='relu')(feature)
    feature=tf.keras.layers.Dense(128,activation='relu')(feature)
    feature=tf.keras.layers.Dropout(0.1)(feature)
    y=tf.keras.layers.Dense(1,activation='sigmoid')(feature)
    TransSSVs_model=tf.keras.models.Model(inputs=inputESM,outputs=y)
    adam=tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08,clipnorm=1.0,clipvalue=0.5)
    TransSSVs_model.compile(loss=[binary_focal_loss(alpha=0.25, gamma=2)],optimizer=adam,metrics=['accuracy',f1_m])
    TransSSVs_model.summary()
    return TransSSVs_model





if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indicator',default='inference')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--weights', default='/model.h5')
    parser.add_argument('--save_dir', default='results/TransSSVs')
    parser.add_argument('--shape_0',default=15,type=int)
    parser.add_argument('--shape_1',default=52,type=int)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--filename', required=True)
    parser.add_argument('--epoch', default=70,required=True,type=int)
    args = parser.parse_args()
    print(args)


    filename=args.filename

    data_only=np.load(args.input_dir+'/data.'+filename+'.npy')
    labels=np.load(args.input_dir+'/label.'+filename+'.npy')
    
    Y_True=labels
    #load model
    TransSSVs_model=get_TransSSVs_model(args.shape_0,args.shape_1)
    TransSSVs_model.load_weights(args.weights)
    Y_Pred=TransSSVs_model.predict(data_only)
    
    Y_Pred_new=[]
    for value in Y_Pred:
        if value<0.5:
            Y_Pred_new.append(0)
        else:
            Y_Pred_new.append(1)
    Y_Pred_new=np.array(Y_Pred_new)
    tn, fp, fn, tp = confusion_matrix(Y_True, Y_Pred_new).ravel()

    print('sensitivity/recall:',tp/(tp+fn))
    print('specificity:',tn/(tn+fp))
    print("F1-score: "+str(f1_score(Y_True,Y_Pred_new)))
    print('false positive rate:',fp/(tn+fp))
    print('false discovery rate:',fp/(tp+fp))
    print('TN:',tn,'FP:',fp,'FN:',fn,'TP:',tp)
    Y_True=Y_True.reshape(Y_True.shape[0],1)
    Y_Pred=Y_Pred.reshape(Y_Pred.shape[0],1)
    Y_Pred_new=Y_Pred_new.reshape(Y_Pred_new.shape[0],1)
    all=np.concatenate((Y_True,Y_Pred,Y_Pred_new),axis=1)
    np.savetxt(args.save_dir+'/y_pred_all.txt',all,fmt='%f',delimiter='\t')
