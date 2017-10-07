import os

PATH = 'config.txt'


def load_train_path():
    if not os.path.exists(PATH) or not os.path.isfile(PATH):
        raise RuntimeError('The path "%s" not found or not file.' % PATH)
    with open(PATH) as f:
        return f.read()


if __name__ == '__main__':
    print(load_train_path())
