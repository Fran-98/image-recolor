from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from network.misc import lab_to_rgb_batch

def infer_and_display(model, image_path, input_size=(256,256)):
    """
    Loads a grayscale image, preprocesses it, runs inference on the trained model using the given condition,
    and displays the resulting colorized image.
    
    Parameters:
      model: The trained U-Net model.
      image_path: Path to the input grayscale image.
      condition_vector: Tensor of shape (1, cond_dim) containing the condition.
      device: Device to run inference on (e.g. "cuda" or "cpu").
      input_size: Desired image size (width, height) as a tuple.
    
    Returns:
      output_pil: The colorized image as a PIL Image.
    """
    # Define the preprocessing transform for the grayscale image.
    preprocess = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),       # Converts to [0,1] tensor with shape (C, H, W)
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    # Load the image, convert it to grayscale and apply the transform.
    img = Image.open(image_path).convert("L")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = preprocess(img).unsqueeze(0).to(device)  # shape: (1, 1, H, W)
    
    # Ensure the model is in evaluation mode.
    model.eval()
    with torch.no_grad():
        # Run inference: the model expects the grayscale image and condition.
        output_tensor = model(input_tensor)
    
    # The output tensor is expected to be (1, 3, H, W). Remove the batch dimension.
    output_tensor = output_tensor.squeeze(0).cpu()  # shape: (3, H, W)
    
    # Convert the tensor to a PIL image.
    to_pil = transforms.ToPILImage()
    output_pil = to_pil(output_tensor)

    # Display both images side by side.
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("Original Grayscale")

    axs[1].imshow(output_pil)
    axs[1].axis("off")
    axs[1].set_title("Colorized Image")

    img = Image.open(image_path.replace('imgs_gray', 'imgs_color'))
    axs[2].imshow(img)
    axs[2].axis("off")
    axs[2].set_title("Original Image")

    plt.show()
    
    return output_pil


def infer_and_display_lab(model, image_path, input_size=(512, 512)):
    """
    Loads a grayscale image, preprocesses it, runs inference on the trained model using LAB space,
    and displays the resulting colorized image.
    
    Parameters:
      model: The trained U-Net model.
      image_path: Path to the input grayscale image.
      input_size: Desired image size (width, height) as a tuple.
    
    Returns:
      output_pil: The colorized image as a PIL Image.
    """
    from PIL import Image
    import torch
    from torchvision import transforms
    import matplotlib.pyplot as plt

    # Use the same transformation as training for the grayscale image.
    preprocess = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # Converts image to [0,1]
    ])

    # Load the image as grayscale
    img_gray = Image.open(image_path).convert("L")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess to get the L channel tensor (shape: (1, 1, H, W))
    L_tensor = preprocess(img_gray).unsqueeze(0).to(device)
    
    # Set model to evaluation mode and predict the ab channels
    model.eval()
    with torch.no_grad():
        # The model expects a tensor like the one used in training
        from network.context import get_yolo_context
        get_yolo_context(L_tensor)
        ab_tensor = model(L_tensor)  # Output shape: (1, 2, H, W)
        print("Predicted ab stats: min =", ab_tensor.min().item(),
              "max =", ab_tensor.max().item(),
              "mean =", ab_tensor.mean().item())

    # Convert the LAB image (L_tensor and predicted ab_tensor) to RGB
    colorized_rgb = lab_to_rgb_batch(L_tensor, ab_tensor)  # Output shape: (1, 3, H, W)
    
    # Convert the tensor to a PIL image for display
    output_pil = transforms.ToPILImage()(colorized_rgb.squeeze(0).cpu())

    # Load the original color image (assumes a similar folder structure as training)
    original_color_path = image_path.replace('imgs_gray', 'imgs_color')
    img_original = Image.open(original_color_path).convert("RGB")

    # Display the grayscale, colorized, and original color images side-by-side.
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(img_gray, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("Original Grayscale")
    
    axs[1].imshow(output_pil)
    axs[1].axis("off")
    axs[1].set_title("Colorized Image")
    
    axs[2].imshow(img_original)
    axs[2].axis("off")
    axs[2].set_title("Original Color Image")
    
    plt.show()

    return output_pil


if __name__ == '__main__':
    
    from unet import UNet
    # Load the model checkpoint.
    checkpoint_path = r'checkpoints\best_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Adjust cond_dim accordingly (e.g., 10 or 512 if using CLIP, etc.)
    cond_dim = 10  
    model = UNet(n_channels=1, n_classes=3, bilinear=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    # Path to your grayscale image.
    image_path = r"dataset\dataset_images\imgs_gray\00000001_(7).jpg"
    
    # Run inference and display the image.
    infer_and_display(model, image_path, device, input_size=(256,256))
