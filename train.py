from network.training_lab import train_model

if __name__ == "__main__":
    train_model(
        dataset_path="dataset/dataset_128",
        output_path="checkpoints_imgnet_6_paisajes",
        n_samples=10000,
        num_epochs=100,
        batch_size=32,
        learning_rate=7e-5,
        weight_decay=1e-6,
        lambda_perceptual=1,#0.02
        lambda_tv=0.00, #1 deja con colores super apagados
        lambda_grad=0.0,
        imgs_size=128,
        pre_trained='checkpoints_imgnet_5_20k/model_epoch_25.pth'
    )