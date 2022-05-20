# it's a script to quantization model to specific data type
import os
import random
import sys
import time

sys.path.append("./")

import random
from glob import glob

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from keras_flops import get_flops
from PIL import Image
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2


class RepresentativeDataset:
    def __init__(self, val_dir, img_size=(96,32), sample_size=200):
        self.val_dir = val_dir
        self.img_size = img_size
        self.sample_size = sample_size
    
    def __call__(self):
        representative_list = random.sample(glob(os.path.join(self.val_dir, "*.jpg")), self.sample_size)
        for image_path in representative_list:
            input_data = Image.open(image_path).convert('L').resize(self.img_size)
            input_data = np.expand_dims(input_data, axis=-1)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = input_data.astype('float32')
            yield [input_data]


def saved_model2pb(
        saved_model_dir,
        input_shape=(1,32,96,1),
        input_node="image",
        output_node="softmax0",
    ):
    # path of the directory where you want to save your model
    frozen_out_path = 'tmp_pb'
    # name of the .pb file
    frozen_graph_filename = "frozen_graph"

    model = keras.models.load_model(
        saved_model_dir,
        custom_objects={'<lambda>': lambda y_true, y_pred: y_pred}
    )
    model = keras.models.Model(
        model.get_layer(name=input_node).input,
        model.get_layer(name=output_node).output,
    )
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(input_shape, name=input_node))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]

    print("-" * 60)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 60)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph to disk
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=frozen_out_path,
        name=f"{frozen_graph_filename}.pb",
        as_text=False
    )
    # Save its text representation
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=frozen_out_path,
        name=f"{frozen_graph_filename}.pbtxt",
        as_text=True
    )

    gf = tfv1.GraphDef()
    m_file = open('{}/frozen_graph.pb'.format(frozen_out_path),'rb')
    gf.ParseFromString(m_file.read())

    with open('{}/somefile.txt'.format(frozen_out_path), 'w+') as the_file:
        for n in gf.node:
            the_file.write(n.name+'\n')

    file = open('{}/somefile.txt'.format(frozen_out_path),'r')
    data = file.readlines()
    output_name = data[len(data)-1].strip()
    print("output name = {}".format(output_name))

    file.seek(0)
    input_name = file.readline().strip()
    print("Input name = {}".format(input_name))

    return frozen_out_path, input_name, output_name


def quantization2tflite(
        model_path,
        mode="pb",
        input_node="image",
        output_node="Identity",
        quantization_mode=tf.uint8,
        save_name="model_uint8",
        representative_dataset=None,
    ):
    assert mode in ["pb", "saved_model"]

    if mode == "saved_model":
        saved_model_dir = model_path
        model = keras.models.load_model(
            saved_model_dir,
            custom_objects={'<lambda>': lambda y_true, y_pred: y_pred}
        )
        model = keras.models.Model(
            model.get_layer(name=input_node).input,
            model.get_layer(name=output_node).output,
        )
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory

    if mode == "pb":
        # Convert the model
        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file = '{}/frozen_graph.pb'.format(model_path),
            input_arrays = [input_node],
            output_arrays = [output_node],
        )

    # only for test
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # for Float16 quantization
    # converter.target_spec.supported_types = [tf.float16]
    converter.representative_dataset = representative_dataset

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


if __name__ == '__main__':
    # width, height
    IMG_SIZE = (96, 32)
    VAL_DIR = 'D:/dataset/license_plate/mini_LPR_dataset/val'
    QUANTIZATION_SAMPLE_SIZE = 200
    MODEL_PATH = 'model_tf_9770'

    quantization_dataset = RepresentativeDataset(VAL_DIR, IMG_SIZE, QUANTIZATION_SAMPLE_SIZE)
    pb_path, input_name, output_name = saved_model2pb(MODEL_PATH)
    quantization2tflite(pb_path, 'pb', input_name, output_name, representative_dataset=quantization_dataset)
