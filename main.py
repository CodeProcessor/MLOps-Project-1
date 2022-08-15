import os

import PIL
from fastapi import FastAPI, UploadFile, File

from pizza.pizza_classification import PizzaClassifier

app = FastAPI()
model_path = "/home/dulanj/Learn/MLOps-Project-1/pizza/resnet18_pizza_cls_model.pth"
pc = PizzaClassifier(model_path)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
):
    input_filename = file.filename
    _file_basename, _file_extension = os.path.splitext(input_filename)
    out_file_path = f"file{_file_extension}"
    contents = file.file.read()
    with open(out_file_path, 'wb') as out_file:
        out_file.write(contents)

    image = PIL.Image.open(out_file_path)
    ret = pc.predict(image)
    return {"message": f"{ret}"}
