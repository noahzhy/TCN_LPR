import tensorflow as tf
import keras


saved_model_dir = "model_tf"
model = keras.models.load_model(
    saved_model_dir,
    custom_objects={'<lambda>': lambda y_true, y_pred: y_pred}
)
model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="softmax0").output
)
# converter = tf.lite.TFLiteConverter.
converter = tf.lite.TFLiteConverter.from_saved_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
# Save the model.
with open('converted_model.tflite', 'wb') as f:
  f.write(tflite_model)
