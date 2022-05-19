import sys
sys.path.append("./")

from models.ms_tcn_9770 import *
import random
import tensorflow as tf
import numpy as np
from keras_flops import get_flops
from PIL import Image
from glob import glob

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
    IMG_SIZE = (96, 32)
    VAL_DIR = 'D:/dataset/license_plate/mini_LPR_dataset/val'
    QUANTIZATION_SAMPLE_SIZE = 200

    def representative_dataset():
        representative_list = random.sample(glob(os.path.join(VAL_DIR, "*.jpg")), QUANTIZATION_SAMPLE_SIZE)
        for image_path in representative_list:
            input_data = Image.open(image_path).convert('L').resize(IMG_SIZE)
            input_data = np.expand_dims(input_data, axis=-1)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = input_data.astype('float32')
            yield [input_data]


    saved_model_dir = "tf_model"
    # model = keras.models.load_model(
    #     saved_model_dir,
    #     custom_objects={'<lambda>': lambda y_true, y_pred: y_pred}
    # )
    # model = keras.models.Model(
    #     model.get_layer(name="image").input,
    #     model.get_layer(name="softmax0").output,
    # )

    # Convert the model
    quantization_mode = tf.uint8
    # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file = 'test/frozen_graph.pb', 
        input_arrays = ['image'],
        output_arrays = ['Identity'],
    )
    # only for test
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.uint8]
    converter.representative_dataset = representative_dataset

    save_name = 'model_uint8'
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.inference_input_type = quantization_mode  # or tf.int8
    converter.inference_output_type = quantization_mode  # or tf.int8
    tflite_model = converter.convert()
    open('{}.tflite'.format(save_name), "wb").write(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_type = interpreter.get_input_details()[0]
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]
    print('output: ', output_type)
