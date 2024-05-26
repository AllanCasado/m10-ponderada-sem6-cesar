from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
from PIL import Image
import io

app = FastAPI()

interpreter = Interpreter(model_path="../modelos/modelo_quantizado_ponderada_sem6_allan_casado.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)  
    image /= 255.0  

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_names[np.argmax(output_data)]

    return JSONResponse(content={"category": predicted_class})
