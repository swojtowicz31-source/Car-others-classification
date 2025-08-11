import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.optimizers import Adam

# Parametry
img_size = 64
batch_size = 32
epochs = 10
imgdir = r'C:...' #prosze podać scieżkę do danych

# Przygotowanie generatorów danych
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    imgdir + '/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    classes=['other', 'car'],
    seed=12345,
    shuffle=True)

val_generator = val_datagen.flow_from_directory(
    imgdir + '/validation',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    classes=['other', 'car'],
    seed=12345,
    shuffle=False)

# Podgląd jednej próbki
Xbatch, Ybatch = next(train_generator)
plt.imshow(Xbatch[4])
print("Etykieta:", Ybatch[4])
plt.show()

# Funkcja tworząca własną CNN
def make_convnet(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn = make_convnet((img_size, img_size, 3))

# Trenowanie modelu
history = cnn.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# Zapis wag modelu
cnn.save_weights('cnn.weights.h5')

# Wykresy dokładności i straty
def plot_history(hist):
    plt.figure(figsize=(12, 5))

    # Dokładność
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='train acc')
    plt.plot(hist.history['val_accuracy'], label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Dokładność')
    plt.legend()

    # Strata
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='train loss')
    plt.plot(hist.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Strata')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# Część 2: Rozszerzanie danych
augmented_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_train_generator = augmented_datagen.flow_from_directory(
    imgdir + '/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    classes=['other', 'car'],
    seed=12345,
    shuffle=True)

# Trenowanie z rozszerzaniem danych
cnn_aug = make_convnet((img_size, img_size, 3))
history_aug = cnn_aug.fit(
    augmented_train_generator,
    validation_data=val_generator,
    epochs=epochs
)

plot_history(history_aug)

# Część 3: Klasyfikacja obrazów VGG-16
vggmodel = VGG16(weights='imagenet', include_top=True)

def classify_with_vgg16(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = preprocess_input(img_arr)
    img_arr = np.expand_dims(img_arr, axis=0)
    preds = vggmodel.predict(img_arr)
    decoded = decode_predictions(preds, top=3)[0]
    print("Predykcje dla obrazu:", image_path)
    for label in decoded:
        print(f"{label[1]}: {label[2]*100:.2f}%")

# Przykład użycia (można zmienic na istniejący plik obrazu)
classify_with_vgg16(os.path.join(imgdir, 'validation/car/0000.jpg')) # mozna zmienic na jakies zdjecie

# Część 4: VGG-16 jako ekstraktor cech
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

def extract_features(generator, sample_count):
    features = np.zeros(shape=(sample_count, 2, 2, 512))  # wyjście VGG16 dla 64x64
    labels = np.zeros(shape=(sample_count,))
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_generator, train_generator.samples)
val_features, val_labels = extract_features(val_generator, val_generator.samples)

train_features = train_features.reshape((train_features.shape[0], -1))
val_features = val_features.reshape((val_features.shape[0], -1))

# Prosty klasyfikator na cechach VGG
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=train_features.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_vgg = model.fit(
    train_features, train_labels,
    epochs=epochs,
    batch_size=32,
    validation_data=(val_features, val_labels)
)

# Wykresy końcowe
plot_history(history_vgg)
