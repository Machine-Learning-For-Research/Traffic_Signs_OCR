from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.optimizers import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
import os

PATH_TRAIN_IMAGES = 'train path'
PATH_VAL_IMAGES = 'validate path'

PATH_WEIGHTS = 'params/weights.h5'
IM_WIDTH = 128
IM_HEIGHT = 128
BATCH_SIZE = 32
CLASSES = len(os.listdir(PATH_TRAIN_IMAGES))
EPOCH = 50
LEARNING_RATE = 1e-2
VAL = True


def calculate_file_num(dir):
    if not os.path.exists(dir):
        return 0
    if os.path.isfile(dir):
        return 1
    count = 0
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        count += calculate_file_num(sub_path)
    return count


def build_generator(path_image, train=True):
    def wrap(value):
        return float(train) and value

    image_generator = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=wrap(15.),
        width_shift_range=wrap(0.2),
        height_shift_range=wrap(0.2),
        shear_range=wrap(0.2),
        zoom_range=wrap(0.2),
        horizontal_flip=train,
        preprocessing_function=None,
    )

    return image_generator.flow_from_directory(
        path_image,
        # classes=['%02d' % i for i in range(CLASSES)],
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
    )


def build_model():
    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu', name='fc1'))
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu', name='fc2'))
    model.add(BatchNormalization())
    model.add(Dense(CLASSES, activation='softmax'))
    return model


if __name__ == '__main__':
    file_num = calculate_file_num(PATH_TRAIN_IMAGES)
    steps_per_epoch = file_num // BATCH_SIZE
    print('Steps number is %d every epoch.' % steps_per_epoch)
    train_generator = build_generator(PATH_TRAIN_IMAGES)
    val_generator = build_generator(PATH_VAL_IMAGES, train=False) if VAL else None

    model = build_model()
    model.summary()

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    if os.path.exists(PATH_WEIGHTS):
        model.load_weights(PATH_WEIGHTS, True)
        print('Load weights.h5 successfully.')
    else:
        print('Model params not found.')

    if not os.path.exists(os.path.dirname(PATH_WEIGHTS)):
        os.makedirs(os.path.dirname(PATH_WEIGHTS))
    try:
        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            callbacks=[
                ModelCheckpoint(PATH_WEIGHTS),
                TensorBoard()
            ],
            epochs=EPOCH,
            validation_data=val_generator if VAL else None,
            validation_steps=calculate_file_num(PATH_VAL_IMAGES) // BATCH_SIZE if VAL else None,
        )
    except KeyboardInterrupt:
        print('\nStop by keyboardInterrupt, try saving weights.')
        model.save_weights(PATH_WEIGHTS)
        print('Save weights successfully.')
