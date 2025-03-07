from network.training_lab import train_model

train_model(
    dataset_path="dataset/dataset_images",
    n_samples=3000,
    num_epochs=10,
    batch_size=8,
    learning_rate=5e-5,
    weight_decay=1e-6,
    lambda_perceptual=0.2,#0.02
    lambda_tv=0.05, #1 deja con colores super apagados
    pre_trained='checkpoints/best_model.pth'
)