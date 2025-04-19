import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPVisionModel
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm
import os
import copy
import multiprocessing

# --- Configuration (Có thể để bên ngoài __main__) ---
DATASET_PATH = "dataset"
MODEL_NAME = "openai/clip-vit-base-patch32" # Hoặc các phiên bản khác như clip-vit-large-patch14
NUM_CLASSES = 4
BATCH_SIZE = 32 # Điều chỉnh tùy theo bộ nhớ GPU
NUM_EPOCHS = 10 # Tăng nếu cần thiết
LEARNING_RATE = 1e-3 # Learning rate cho lớp linear
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_SAVE_PATH = "best_mushroom_classifier_head.pth"


def main(): # Đưa logic chính vào hàm main
    print(f"Using device: {DEVICE}")

    # --- Class Mapping (Quan trọng!) ---
    # Đảm bảo thứ tự này khớp với yêu cầu
    # ImageFolder sẽ sắp xếp theo alphabet nếu không chỉ định class_to_idx
    # Thứ tự alphabet: ['Đùi gà Baby (cắt ngắn)', 'bào ngư xám + trắng', 'linh chi trắng', 'nấm mỡ']
    # Ánh xạ mong muốn: 0: nấm mỡ, 1: bào ngư xám + trắng, 2: Đùi gà Baby (cắt ngắn), 3: linh chi trắng
    class_names_sorted = sorted(os.listdir(os.path.join(DATASET_PATH, "train")))
    print(f"Detected class names (sorted): {class_names_sorted}")

    # Tạo ánh xạ thủ công để đảm bảo đúng nhãn
    class_to_idx = {
        "nấm mỡ": 0,
        "bào ngư xám + trắng": 1,
        "Đùi gà Baby (cắt ngắn)": 2,
        "linh chi trắng": 3
    }
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"Using class mapping: {class_to_idx}")


    # --- Load CLIP Processor and Define Transforms ---
    try:
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading CLIP processor: {e}")
        print("Make sure you have internet connection and the model name is correct.")
        exit()

    # Lấy các thông số chuẩn hóa từ processor
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std
    # Sửa lỗi KeyError: 'height'. Thường dùng 'shortest_edge' cho CLIP.
    image_size = processor.image_processor.size["shortest_edge"] 

    # Transforms hiện đại cho training
    train_transforms = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)), # Cắt ngẫu nhiên với kích thước gần gốc
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(), # Augmentation tự động, mạnh mẽ
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ])

    # Transforms chuẩn cho validation/test (chỉ resize, crop giữa, normalize)
    val_transforms = T.Compose([
        T.Resize(image_size), # Resize cạnh nhỏ nhất tới image_size
        T.CenterCrop(image_size), # Crop phần giữa
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ])

    # --- Load Data ---
    try:
        train_dataset = ImageFolder(
            root=os.path.join(DATASET_PATH, "train"),
            transform=train_transforms
        )
        # Gán lại class_to_idx để đảm bảo đúng thứ tự
        train_dataset.class_to_idx = class_to_idx
        train_dataset.classes = [idx_to_class[i] for i in range(NUM_CLASSES)]

        val_dataset = ImageFolder(
            root=os.path.join(DATASET_PATH, "val"),
            transform=val_transforms
        )
        # Gán lại class_to_idx
        val_dataset.class_to_idx = class_to_idx
        val_dataset.classes = [idx_to_class[i] for i in range(NUM_CLASSES)]

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Train class to index mapping: {train_dataset.class_to_idx}")
        print(f"Validation class to index mapping: {val_dataset.class_to_idx}")


        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATASET_PATH}. Please ensure the 'dataset' folder exists with 'train' and 'val' subdirectories.")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

    # --- Load CLIP Vision Model (Frozen) ---
    try:
        clip_vision_model = CLIPVisionModel.from_pretrained(MODEL_NAME).to(DEVICE)
    except Exception as e:
        print(f"Error loading CLIP vision model: {e}")
        exit()

    # Đóng băng tất cả các tham số của CLIP
    for param in clip_vision_model.parameters():
        param.requires_grad = False

    clip_vision_model.eval() # Đặt ở chế độ đánh giá

    # --- Define Classification Head ---
    embedding_dim = clip_vision_model.config.hidden_size
    classifier_head = nn.Linear(embedding_dim, NUM_CLASSES).to(DEVICE)

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss()
    # Chỉ tối ưu hóa các tham số của classifier_head
    optimizer = optim.AdamW(classifier_head.parameters(), lr=LEARNING_RATE)
    # Scheduler (tùy chọn nhưng hữu ích)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader))

    # --- Training Loop ---
    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        # Training Phase
        classifier_head.train() # Đặt head ở chế độ train
        running_loss = 0.0
        train_preds = []
        train_labels = []

        progress_bar_train = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in progress_bar_train:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Lấy đặc trưng từ CLIP (không cần tính gradient)
            with torch.no_grad():
                # Lấy pooler_output hoặc last_hidden_state[:, 0] tùy thuộc vào cách CLIP trả về
                # Thường thì pooler_output là đại diện tốt cho ảnh
                image_features = clip_vision_model(pixel_values=inputs).pooler_output

            # Đưa đặc trưng qua lớp linear head
            outputs = classifier_head(image_features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step() # Cập nhật LR sau mỗi batch

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            progress_bar_train.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = accuracy_score(train_labels, train_preds)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation Phase
        classifier_head.eval() # Đặt head ở chế độ eval
        val_loss = 0.0
        val_preds = []
        val_labels = []

        progress_bar_val = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad(): # Không cần tính gradient khi đánh giá
            for inputs, labels in progress_bar_val:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                image_features = clip_vision_model(pixel_values=inputs).pooler_output
                outputs = classifier_head(image_features)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                progress_bar_val.set_postfix(loss=loss.item())


        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = accuracy_score(val_labels, val_preds)
        print(f"Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Lưu model tốt nhất
        if val_epoch_acc > best_val_accuracy:
            print(f"Validation accuracy improved ({best_val_accuracy:.4f} -> {val_epoch_acc:.4f}). Saving model...")
            best_val_accuracy = val_epoch_acc
            # Chỉ lưu trọng số của lớp classifier head
            best_model_state = copy.deepcopy(classifier_head.state_dict())
            torch.save(best_model_state, BEST_MODEL_SAVE_PATH)
            print(f"Best model head saved to {BEST_MODEL_SAVE_PATH}")

    # --- Final Evaluation ---
    print("\n--- Final Evaluation on Validation Set ---")
    if best_model_state:
        classifier_head.load_state_dict(best_model_state)
        print(f"Loaded best model head from {BEST_MODEL_SAVE_PATH} with accuracy: {best_val_accuracy:.4f}")
    else:
         print("No best model was saved during training. Evaluating with the final model.")

    classifier_head.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Final Evaluation"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            image_features = clip_vision_model(pixel_values=inputs).pooler_output
            outputs = classifier_head(image_features)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # In báo cáo chi tiết
    target_names = [idx_to_class[i] for i in range(NUM_CLASSES)] # Lấy tên lớp theo đúng thứ tự 0, 1, 2, 3
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print("\nClassification Report:")
    print(report)

    print(f"Final accuracy on validation set: {accuracy_score(all_labels, all_preds):.4f}")

    print("\nTraining finished.")
    print(f"To use the model, load the CLIP vision model ('{MODEL_NAME}') and the trained classifier head from '{BEST_MODEL_SAVE_PATH}'.") 

if __name__ == '__main__':
    multiprocessing.freeze_support() # Cần thiết cho Windows khi đóng gói hoặc dùng spawn
    main() # Gọi hàm main 