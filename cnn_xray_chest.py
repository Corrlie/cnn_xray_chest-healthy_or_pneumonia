import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
cst_batch = 16

size = 250

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range = 0.2)
train_set = train_datagen.flow_from_directory('train', batch_size=cst_batch, shuffle=True, target_size=(size,size), interpolation='bicubic', class_mode='binary', color_mode='grayscale')


val_datagen = ImageDataGenerator(rescale=1./255, zoom_range = 0.2)
val_set = val_datagen.flow_from_directory('val', batch_size=cst_batch, shuffle=True, target_size=(size,size), interpolation='bicubic', class_mode='binary', color_mode='grayscale')

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(50, (3, 3), activation='relu', input_shape=(size, size, 1)))
cnn.add(tf.keras.layers.AveragePooling2D((2, 2)))

cnn.add(tf.keras.layers.Conv2D(100, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.AveragePooling2D((2, 2)))


cnn.add(tf.keras.layers.Conv2D(150, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.AveragePooling2D((2, 2)))
cnn.add(tf.keras.layers.Dropout(0.3))

cnn.add(tf.keras.layers.Conv2D(200, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.AveragePooling2D((2, 2)))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(200, activation='relu', activity_regularizer=tf.keras.regularizers.l2(0.001)))
cnn.add(tf.keras.layers.Dense(50, activation='relu', activity_regularizer=tf.keras.regularizers.l2(0.001)))
cnn.add(tf.keras.layers.Dropout(0.4))
cnn.add(tf.keras.layers.Dense(1, activation='sigmoid', activity_regularizer=tf.keras.regularizers.l2(0.001)))
cnn.summary()
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = cnn.fit_generator(train_set, validation_data=val_set, epochs=20)

model_json = cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

cnn.save_weights("model.h5")

