from config import *
from data_produce import *
from model import multi_gpu_model
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
def main():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    config_id = get_config(is_train=False)
    dense_model = multi_gpu_model(sess=sess, gpu_list=[0, 1], config=config_id)
    dense_model.restore(config_id.ckpt_dir)

    data_generator = data_produce(is_train=True)
    x, y = data_generator.load_data()
    minibatches = data_generator.minibatches_produce(x, y, seed=0)
    for minibatch in minibatches:
        batch_x,batch_y=minibatch
        _loss,_psnr,_mse=dense_model.deploy(batch_x,batch_y,train_phase=False)
        print(_loss)
if __name__=='__main__':
    main()



