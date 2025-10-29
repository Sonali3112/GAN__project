epochs = 100000
batch_size = 64

for epoch in range(epochs+1):
   
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_images = generator_cnn.predict(noise)

    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator_cnn.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator_cnn.train_on_batch(generated_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, 100))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan_cnn.train_on_batch(noise, valid_labels)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss: {d_loss[0]}, G Loss: {g_loss}")

    if epoch % 1000 == 0:
        test_noise = np.random.normal(0, 1, (1, 100))
        test_img = generator_cnn.predict(test_noise)[0].reshape(28, 28)
        plt.imshow(test_img, cmap='gray')
        plt.axis('off')
        plt.show()
