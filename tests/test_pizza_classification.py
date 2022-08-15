"""
Copyright (C) CUBE Content Governance Global Limited - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Dulan Jayasuriya <dulan.jayasuriya@cube.global>, 15 August 2022
"""
import PIL

from pizza.pizza_classification import PizzaClassifier


def test_predict():
    pc = PizzaClassifier("/home/dulanj/Learn/MLOps-Project-1/pizza/resnet18_pizza_cls_model.pth")
    image = PIL.Image.open("/home/dulanj/Learn/MLOps-Project-1/dataset/pizza_not_pizza_split/test/pizza/29417.jpg")
    ret = pc.predict(image)

    assert isinstance(ret, tuple)
    assert len(ret) == 2
