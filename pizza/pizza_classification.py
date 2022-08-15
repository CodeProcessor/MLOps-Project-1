import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from pizza.custom_logging import logger
from pizza.singleton import Singleton

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PizzaClassifier(metaclass=Singleton):
    CLASSES = ["Pizza", "Non-Pizza"]

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.getenv("PIZZA_MODEL_PATH", "resnet18_pizza_cls_model.pth")
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.threshold = 0.5

    def load_model(self, model_path):
        logger.info(f"Loading Style Detection Model from {model_path}")
        _loading_start = time.time()
        model = self.get_model()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model = model.to(device)
        logger.info(f"Model loading took : {round(time.time() - _loading_start, 3)} seconds")
        return model

    def get_model(self):
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft = model_ft.to(device)
        # logger.debug(summary(model_ft, (3, 224, 224)))
        return model_ft

    def _pre_process(self, image):
        return self.transform(image)

    def _post_process(self, prediction):
        pred = np.squeeze(prediction)
        arg_max = np.argmax(pred)
        _confidence = pred[arg_max]
        if _confidence > self.threshold:
            _class = self.CLASSES[arg_max]
        else:
            _class = "Unknown"
            _confidence = 0
        return _class, _confidence

    def _inference(self, image):
        image_device = image.to(device)
        image_unz = image_device.unsqueeze(0)
        pred = self.model(image_unz)
        m = nn.Softmax(dim=1)
        output = m(pred)
        ret = output.detach().cpu().numpy()
        return ret

    def predict(self, image):
        start_time = time.time()
        image = self._pre_process(image)
        pred = self._inference(image)
        result = self._post_process(pred)
        logger.info(f"Style model took {round(1000 * (time.time() - start_time), 2)} ms")
        return result
