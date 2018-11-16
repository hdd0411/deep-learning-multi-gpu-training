class Config(object):
    pass
def get_config(is_train):
    config=Config()
    if is_train:
        config.batch_size=64
        config.lr=1e-4
        config.epoch=300
        config.tmp_dir='tmp'
        config.ckpt_path='/home/a/Downloads/generate_data/save_model'
        config.data_path=r'/home/a/Downloads/generate_data'
        config.filename='48train_N_0.5.h5'
        config.nb_blocks=4
        config.filters=10
        config.dropout_rate=0.2
        config.weight_decay=0.00001
    else:
        config.batch_size=10
        config.result_dir='result'
        config.ckpt_dir='/home/a/Downloads/generate_data/save_model'
        config.data_path='/home/a/Downloads/generate_data'
        config.filename='48test_N_0.5.h5'
        config.nb_blocks = 4
        config.filters = 10
        config.dropout_rate = 0.2
        config.weight_decay = 0.00001
    return config

