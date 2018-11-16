from ops import *
class DenseNet():
    def __init__(self, nb_blocks, filters, dropout_rate,training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.dropout_rate=dropout_rate
        self.training = training



    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization(x, training=self.training, scope=scope+'_batch1')
            x = relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[3,3], layer_name=scope+'_conv1')
            x = drop_out(x, rate=self.dropout_rate, training=self.training)

            x = batch_normalization(x, training=self.training, scope=scope+'_batch2')
            x = relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = drop_out(x, rate=self.dropout_rate, training=self.training)
            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization(x, training=self.training, scope=scope+'_batch1')
            x = relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = drop_out(x, rate=self.dropout_rate, training=self.training)
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)
            x = concatenation(layers_concat)
            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[3,3], stride=1, layer_name='conv0')

        x = self.dense_block(input_x=x, nb_layers=5, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=5, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=5, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=5, layer_name='dense_final')
        x = batch_normalization(x, training=self.training, scope='linear_batch')
        x = relu(x)
        x = conv_layer(x, filter=4 * self.filters, kernel=[3, 3], stride=1, layer_name='conv1')
        x = conv_layer(x, filter=1, kernel=[3, 3], stride=1, layer_name='conv2')

        return x



class multi_gpu_model(object):
    def __init__(self,sess,gpu_list,config):
        self.sess=sess
        self.gpu_list=gpu_list
        self.is_train=tf.placeholder(tf.bool,name='training_flag')
        self.nb_blocks = config.nb_blocks
        self.filters = config.filters
        self.dropout_rate = config.dropout_rate
        self.learning_rate=tf.placeholder(tf.float32, shape=[])
        self.opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        self.weight_decay=config.weight_decay
        self.models = []
        self.densenet=DenseNet(nb_blocks=self.nb_blocks, filters=self.filters, dropout_rate=self.dropout_rate,training=self.is_train)


    #def deal_multi_gpu(self):
        num_gpu=len(self.gpu_list)
        with tf.device('/cpu:0'):
            print('build model...')

            print('build model on gpu tower...')



            for gpu_id in range(num_gpu):
                with tf.device('/gpu:%d' % gpu_id):
                    print('tower:%d...' % gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id) as scope:
                        with tf.variable_scope('cpu_variables', reuse=(gpu_id) > 0):
                            x = tf.placeholder(tf.float32, (None, None, None, 1), name='x')
                            y = tf.placeholder(tf.float32, (None, None, None, 1), name='y')

                           ## replace different models
                            pred=self.densenet.Dense_net(x)

                            mse_loss = loss_cost(pred, y)
                            psnr = PSNR_cal(pred, y)
                            tf.add_to_collection("losses", mse_loss)
                            ###l2 regularier####
                            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
                            total_loss = tf.add_n(tf.get_collection("losses", scope)) + l2_loss * self.weight_decay
                            grads = self.opt.compute_gradients(total_loss)

                            ##gradient clip###########
                            for i, (g, v) in enumerate(grads):
                                if g is not None:
                                    grads[i] = (tf.clip_by_norm(g, 0.1), v)

                            ## gradient clip finish ##
                            self.models.append((x, y, pred, total_loss, psnr, mse_loss, grads))
            print('build model on gpu tower done.')
            print('reduce model on cpu...')
            tower_x, tower_y, tower_preds, tower_losses, tower_psnr, tower_mse, tower_grads = zip(*self.models)
            self.aver_loss_op = tf.reduce_mean(tower_losses)
            self.aver_psnr_op = tf.reduce_mean(tower_psnr)
            self.aver_mse_op = tf.reduce_mean(tower_mse)
            self.apply_gradient_op = self.opt.apply_gradients(average_gradients(tower_grads))
        self.saver = tf.train.Saver()

    def fit(self,batch_x,batch_y,lr,train_phase):
        self.sess.run(tf.global_variables_initializer())
        batch_size=batch_x.shape[0]
        payload_per_gpu=batch_size/len(self.gpu_list)
        inp_dict={}
        inp_dict[self.learning_rate]=lr
        inp_dict[self.is_train]=train_phase
        inp_dict = feed_all_gpu(inp_dict, self.models, payload_per_gpu,batch_x,batch_y)
        _, _loss, _psnr, _mse = self.sess.run([self.apply_gradient_op, self.aver_loss_op, self.aver_psnr_op, self.aver_mse_op], inp_dict)

        return _loss,_psnr,_mse

    def deploy(self,x,y,train_phase):
        batch_size = x.shape[0]
        payload_per_gpu = batch_size / len(self.gpu_list)
        inp_dict = {}
        inp_dict[self.is_train] = train_phase
        inp_dict = feed_all_gpu(inp_dict, self.models, payload_per_gpu,x, y)
        test_total_loss, test_psnr_loss, test_mse_loss = self.sess.run([self.aver_loss_op, self.aver_psnr_op, self.aver_mse_op], inp_dict)
        return test_total_loss, test_psnr_loss, test_mse_loss
    def save(self,ckpt_path):
        self.saver.save(self.sess,ckpt_path+'/model.ckpt')
    def restore(self,ckpt_path):
        self.saver.restore(self.sess,ckpt_path+'/model.ckpt')




























