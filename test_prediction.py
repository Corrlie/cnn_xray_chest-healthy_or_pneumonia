from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
file = open('model.json', 'r')
loaded = file.read()
file.close()

loaded_model = model_from_json(loaded)
loaded_model.load_weights("model.h5")
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

loaded_model.summary()

cst_batch = 32
size = 250

test_datagen = ImageDataGenerator(rescale=1./255)
predict_set = test_datagen.flow_from_directory('test', batch_size=cst_batch, target_size=(size,size), interpolation='bicubic', class_mode='binary', seed=1, color_mode='grayscale')


x_pred, y_pred = predict_set.next()
pr_cl = loaded_model.predict_classes(x_pred)

pr_mean_score = 0.0

for element in range (predict_set.__len__()):
    pr_im_all, pr_lab_all = predict_set.next()
    pr_score_all = loaded_model.evaluate(pr_im_all, pr_lab_all)
    pr_mean_score += pr_score_all[1]
pr_mean_all = pr_mean_score/(predict_set.__len__())
print(f"Prediction acc for all pred img: {pr_mean_all}")