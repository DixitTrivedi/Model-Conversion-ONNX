## IMPORT LIBRARIES
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle

model = pickle.load(open('model/model.pkl', 'rb'))
initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(model, initial_types=initial_type)

with open('model/model.onnx', 'wb') as f:
    f.write(onx.SerializeToString())

