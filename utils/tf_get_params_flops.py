import sys
sys.path.append("./")

from models.model import *

import tensorflow as tf
from keras_flops import get_flops


# def get_flops(model):
#     graph = tf.compat.v1.get_default_graph()
#     run_meta = tf.compat.v1.RunMetadata()
#     opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#     param_stats = tf.compat.v1.profiler.profile(
#         tf.compat.v1.get_default_graph(),
#         cmd='code',
#         options=opts
#     )
#     flops = tf.compat.v1.profiler.profile(
#         graph=graph,
#         run_meta=run_meta,
#         cmd='op',
#         options=opts
#     )
#     # time_and_memory = tf.profiler.profile(
#     #     tf.get_default_graph(),
#     #     run_meta=run_meta,
#     #     cmd='op',
#     #     options=tf.profiler.ProfileOptionBuilder.time_and_memory()
#     # )
#     print('Total_params: %f (M)Params' % float(param_stats.total_parameters / 1000000))
#     print('Total_flops: %f (G)FLOPs' % float(flops.total_float_ops / 1000000))
#     # print('Total_params: %d\n' % time_and_memory.total_time_and_memory)


if __name__ == '__main__':
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL), batch_size=BATCH_SIZE, name='input_image')
    # inputs = STN()(inputs)
    model = TCN_LPR()
    model.summary()
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
