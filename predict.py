 
import os
import torch
from torchvision import transforms
from PIL import Image
from model import MyCustomModel
import config

def cryptic_inf_f(data_folder_path):
    DEVICE = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # Fixed class names
    class_names = ['buildings', 'street', 'mountain', 'sea', 'forest', 'glacier']

    model = MyCustomModel(num_classes=len(class_names))
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    inference_transform = transforms.Compose([
        transforms.Resize((config.resize_x, config.resize_y)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    image_paths = []
    valid_exts = (".jpg", ".jpeg", ".png")

    # Collect all image paths from the folder and subfolders
    for root, _, files in os.walk(data_folder_path):
        for file in files:
            if file.lower().endswith(valid_exts):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    if not image_paths:
        print("No valid images found!")
        return []

    img_list = []
    img_names = []

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = inference_transform(img)
        img_list.append(img)
        img_names.append(os.path.basename(path))  # only the file name

    batch = torch.stack(img_list).to(DEVICE)

    predictions = []
    with torch.no_grad():
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    for img_name, pred_idx in zip(img_names, preds):
        label = class_names[pred_idx]
        predictions.append((img_name, label))

    return predictions

# Example Usage
if __name__ == "__main__":
    predictions = cryptic_inf_f("data/")
    for img_name, label in predictions:
        print(f"{img_name}: {label}")