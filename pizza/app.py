import PIL

from pizza.pizza_classification import PizzaClassifier


def main():
    pc = PizzaClassifier()
    image = PIL.Image.open("/home/dulanj/Learn/MLOps-Project-1/dataset/pizza_not_pizza_split/test/pizza/29417.jpg")
    ret = pc.predict(image)
    print(ret)


if __name__ == '__main__':
    main()
