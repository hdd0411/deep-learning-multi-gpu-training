import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network
def batch_normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))
def drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)
def relu(x):
    return tf.nn.relu(x)
def average_pooling(x, pool_size=[2,2], stride=1, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
def max_pooling(x, pool_size=[3,3], stride=1, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
def concatenation(layers) :
    return tf.concat(layers, axis=3)


### loss function define
def loss_cost(x,y):
    result=1/2*tf.reduce_mean(tf.square(x-y))
    return result
def loss_cost_l1(x,y):
    result=tf.reduce_mean(tf.abs(x-y))
    return result
def PSNR_cal(result,Y):
    erro=result-Y
    mse=tf.reduce_mean(tf.square(erro))
    psnr = 10.0*tf.log(255.0*255.0/(mse+1e-8))/tf.log(10.0)
    return psnr


## multi_gpu funcction
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y):
    for i in range(len(models)):
        x, y, _, _, _, _, _ = models[i]
        start_pos = int(i * payload_per_gpu)
        stop_pos = int((i + 1) * payload_per_gpu)
        inp_dict[x] = batch_x[start_pos:stop_pos]
        inp_dict[y] = batch_y[start_pos:stop_pos]
    return inp_dict


