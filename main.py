# =============================
# 1. Importar dependências
# =============================
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# =============================
# 2. Carregar base de dados LFW (Labeled Faces in the Wild)
# =============================
(ds_train, ds_val), ds_info = tfds.load(
    'lfw',  # dataset de rostos
    split=['train[:80%]', 'train[80%:]'],
    with_info=True
)

NUM_CLASSES = ds_info.features['label'].num_classes
print("Número de classes:", NUM_CLASSES)

# =============================
# 3. Pré-processamento
# =============================
IMG_SIZE = 160

def preprocess(example):
    image = example["image"]
    label = example["label"]
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

BATCH_SIZE = 32

train = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val = ds_val.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =============================
# 4. Modelo pré-treinado (Transfer Learning com MobileNetV2)
# =============================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # congela os pesos da base

# =============================
# 5. Construir modelo de reconhecimento facial
# =============================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================
# 6. Treinar modelo
# =============================
history = model.fit(
    train,
    validation_data=val,
    epochs=5
)

# =============================
# 7. Avaliar resultados
# =============================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(acc, label="Treino")
plt.plot(val_acc, label="Validação")
plt.legend()
plt.title("Acurácia")

plt.subplot(1,2,2)
plt.plot(loss, label="Treino")
plt.plot(val_loss, label="Validação")
plt.legend()
plt.title("Loss")

plt.show()
