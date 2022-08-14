import os
import shutil


class DataPreperation:
    """
    This class is used to prepare the data for the model
    """
    SPLIT_RATIO = {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    }

    def __init__(self, source, destination):
        self.source = source
        if os.path.exists(destination):
            shutil.rmtree(destination)
        os.makedirs(destination)
        self.destination = destination

        self.classes = {}
        self.set_counts()

    def set_counts(self):
        for dir in os.listdir(self.source):
            _full_dir = os.path.join(self.source, dir)
            if os.path.isdir(_full_dir):
                if dir not in self.classes:
                    self.classes[dir] = len(os.listdir(_full_dir))
                    print("Class: ", dir, "Count: ", self.classes[dir])

    def data_splitter(self):
        for _dir in self.classes:
            _source_dir = os.path.join(self.source, _dir)
            _no_of_images_in_class = len(os.listdir(_source_dir))
            for _key, _value in self.SPLIT_RATIO.items():
                _no_of_images_to_move = int(_value * _no_of_images_in_class)
                _moved = 0
                for file in os.listdir(_source_dir):
                    _moved += 1
                    _destination_dir = os.path.join(self.destination, _key, _dir)
                    if _moved >= _no_of_images_to_move:
                        break
                    if not os.path.exists(_destination_dir):
                        os.makedirs(_destination_dir)
                    _src = os.path.join(_source_dir, file)
                    _dst = os.path.join(_destination_dir, file)
                    shutil.copy(_src, _dst)
                    # print("{} -> {}".format(_src, _dst))
                print("Moved {} images to {}/{}".format(_moved, _dir, _key))


if __name__ == '__main__':
    dp = DataPreperation(
        source="pizza_not_pizza",
        destination="pizza_not_pizza_split"
    )
    dp.data_splitter()
