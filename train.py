from config import *
from data_produce import *
from model import multi_gpu_model
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
def main():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess=tf.Session(config=config)

    config_id=get_config(is_train=True)
    lr_=config_id.lr
    epoch=config_id.epoch

    dense_model=multi_gpu_model(sess=sess,gpu_list=[0,1],config=config_id)

    data_generator = data_produce(is_train=True)
    x, y = data_generator.load_data()
    minibatches = data_generator.minibatches_produce(x, y, seed=1)
    minibatches=minibatches[0:10]
    for i in range(epoch):
        for minibatch in minibatches:
            batch_x,batch_y=minibatch
            train_loss,train_psnr,train_mse=dense_model.fit(batch_x,batch_y,lr=lr_,train_phase=True)
            print(train_loss)
        if i%30==0:
            lr_=max(lr_*0.1,0.00001)
        if i%2==0:
            dense_model.save(config_id.ckpt_path)
if __name__=='__main__':
    main()




