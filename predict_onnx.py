import onnxruntime as rt
import pickle
import numpy as np

sc = pickle.load(open('model/scalar.pkl', 'rb'))
data = sc.transform([[5.1, 3.5 ,1.4, 0.2]])

sess = rt.InferenceSession("model/model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onnx = sess.run([label_name], {input_name: data.astype(np.float32)})[0]
print(pred_onnx)