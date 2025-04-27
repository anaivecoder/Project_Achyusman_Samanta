# import os
# import torch
# from torchvision import transforms
# from PIL import Image
# from model import MyCustomModel
# import config

# def Inference_p():
#     DEVICE = torch.device(config.device if torch.cuda.is_available() else "cpu")
#     print("Using device:", DEVICE)

#     transform = transforms.Compose([
#         transforms.Resize((config.resize_x, config.resize_y)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])

#     from torchvision.datasets import ImageFolder
#     class_names = ImageFolder("my_dataset/train").classes

#     model = MyCustomModel(num_classes=len(class_names))
#     model.load_state_dict(torch.load(config.checkpoint_path, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()

#     predictions = []
#     image_dir = "data"
#     valid_exts = (".jpg", ".jpeg", ".png")

#     image_paths = [
#         os.path.join(dp, f)
#         for dp, _, filenames in os.walk(image_dir)
#         for f in filenames if f.lower().endswith(valid_exts)
#     ]

#     for path in image_paths:
#         img = Image.open(path).convert("RGB")
#         img = transform(img).unsqueeze(0).to(DEVICE)
#         with torch.no_grad():
#             output = model(img)
#             _, pred = torch.max(output, 1)
#             predictions.append((path, class_names[pred.item()]))

#     return predictions

# print(Inference_p())

import os
import torch
from torchvision import transforms
from PIL import Image
from model import MyCustomModel
import config

def cryptic_inf_f(data_folder_path):
    DEVICE = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    from torchvision.datasets import ImageFolder
    class_names = ImageFolder("my_dataset/train").classes

    model = MyCustomModel(num_classes=len(class_names))
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    inference_transform = transforms.Compose([
        transforms.Resize((config.resize_x, config.resize_y)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img_list = []
    image_paths = []

    valid_exts = (".jpg", ".jpeg", ".png")

    # Go through all subfolders and collect images
    for root, _, files in os.walk(data_folder_path):
        for file in files:
            if file.lower().endswith(valid_exts):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = inference_transform(img)
        img_list.append(img)

    if not img_list:
        print("No valid images found!")
        return []

    batch = torch.stack(img_list).to(DEVICE)

    with torch.no_grad():
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)

    labels = [class_names[p.item()] for p in preds]

    return labels

# Give the data folder path
predictions = cryptic_inf_f("data/")
print(predictions)
