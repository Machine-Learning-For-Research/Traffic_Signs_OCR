from properties import Properties

CONFIG_PATH = 'config.properties'
properties = Properties(CONFIG_PATH)


def load_train_path():
    return properties.get('trainPath')


def load_test_path():
    return properties.get('testPath')


if __name__ == '__main__':
    print(load_train_path())
    print(load_test_path())
