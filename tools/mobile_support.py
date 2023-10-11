import keras
import tensorflow as tf

net = keras.models.load_model("./results/trained_models/meomeo.hdf5")
import tf2onnx
import onnx

# keras.save_model.save

spec = [tf.TensorSpec((None, 21 * 2), dtype=tf.double, name="input_1")]
onnx_model, _ = tf2onnx.convert.from_keras(net, input_signature=spec, opset=13)
onnx.save_model(onnx_model, "./results/trained_models/meomeo.onnx")
