# import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Load and preprocess the dataset
df = pd.read_csv('Cardiovascular_Disease_Dataset.csv') # Replace with your dataset path
features = df.iloc[:, :-1].values # Features (13 columns)
labels = df.iloc[:, -1].values # Binary target (0 or 1)
scaler = StandardScaler()
features = scaler.fit_transform(features) # Normalize features
features = df.values.astype('float32')
features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))
# Hyperparameters
batch_size = 32
noise_dim = 16
epochs = 1000
lambda_gp = 10 # Gradient penalty coefficient
learning_rate = 1e-4
64
# Build the generator
def build_generator():
model = tf.keras.Sequential([
tf.keras.layers.Dense(64, activation='relu', input_dim=noise_dim),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(features.shape[1]) # Output matches feature dimensions
])
return model
# Build the discriminator
def build_discriminator():
model = tf.keras.Sequential([
tf.keras.layers.Dense(128, activation='relu', input_dim=features.shape[1]),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(1) # Output a single scalar for WGAN
])
return model
# Gradient penalty function
def gradient_penalty(discriminator, real_data, fake_data):
batch_size = tf.shape(real_data)[0]
65
alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0, dtype=tf.float32)
interpolated = alpha * real_data + (1 - alpha) * fake_data
with tf.GradientTape() as tape:
tape.watch(interpolated)
pred = discriminator(interpolated, training=True)
grads = tape.gradient(pred, interpolated)
grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
return tf.reduce_mean((grads_norm - 1.0) ** 2)
# Instantiate models
generator = build_generator()
discriminator = build_discriminator()
def weight_init(shape, dtype=None):
return tf.random.normal(shape, stddev=0.02, dtype=dtype)
# Apply to each layer
generator.layers[0].kernel_initializer = weight_init
discriminator.layers[0].kernel_initializer = weight_init
# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5, beta_2=0.9)
# Training step
@tf.function
def train_step(real_data):
batch_size = tf.shape(real_data)[0]
noise = tf.random.normal([batch_size, noise_dim], dtype=tf.float32)
66
# Train Discriminator
with tf.GradientTape() as disc_tape:
fake_data = generator(noise, training=True)
real_output = discriminator(real_data, training=True)
fake_output = discriminator(fake_data, training=True)
gp = gradient_penalty(discriminator, real_data, fake_data)
disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + lambda_gp *
gp
disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
discriminator_optimizer.apply_gradients(zip(disc_gradients,
discriminator.trainable_variables))
# Train Generator
with tf.GradientTape() as gen_tape:
fake_data = generator(noise, training=True)
fake_output = discriminator(fake_data, training=True)
gen_loss = -tf.reduce_mean(fake_output)
gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
return gen_loss, disc_loss
# Training loop
def train(dataset, epochs):
for epoch in range(epochs):
67
for i in range(0, len(dataset), batch_size):
real_data = dataset[i:i + batch_size]
real_data = tf.convert_to_tensor(real_data, dtype=tf.float32) # Ensure float32
gen_loss, disc_loss = train_step(real_data)
if epoch % 100 == 0:
print(f"Epoch {epoch}/{epochs}, Generator Loss: {gen_loss.numpy()},
Discriminator Loss: {disc_loss.numpy()}")
# Train the WGAN
train(features, epochs)
# Assuming the 'generator' model is already trained
# 'noise_dim' should match the input size of your generator model
def generate_synthetic_data(generator, num_samples, noise_dim):
# Generate random noise input for the generator
noise = tf.random.normal([num_samples, noise_dim], dtype=tf.float32)
# Generate synthetic data
synthetic_data = generator(noise, training=False).numpy()
# Convert synthetic data to a DataFrame
synthetic_df = pd.DataFrame(synthetic_data, columns=[f'feature_{i+1}' for i in
range(synthetic_data.shape[1])])
return synthetic_df
68
# Example: Generate 500 synthetic samples
synthetic_data_df = generate_synthetic_data(generator, num_samples=500,
noise_dim=noise_dim)
