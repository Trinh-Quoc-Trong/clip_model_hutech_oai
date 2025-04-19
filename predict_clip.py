import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image
import os
import csv
from tqdm.auto import tqdm

# Cấu hình
MODEL_NAME = "openai/clip-vit-base-patch32"
BEST_MODEL_PATH = "best_mushroom_classifier_head.pth"
TEST_FOLDER = "D:/code/projects/clip_hutech_oai/test"  # Đường dẫn tuyệt đối
OUTPUT_FILE = "submission.csv"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

# Định nghĩa ánh xạ lớp (giống như trong training)
idx_to_class = {
    0: "nấm mỡ",
    1: "bào ngư xám + trắng",
    2: "Đùi gà Baby (cắt ngắn)",
    3: "linh chi trắng"
}

# Dataset tùy chỉnh để đọc ảnh test
class MushroomTestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = []
        self.image_ids = []
        
        # Lấy tất cả file ảnh từ thư mục test
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(image_folder, filename))
                # Lấy ID ảnh (bỏ phần mở rộng file)
                img_id = os.path.splitext(filename)[0]
                self.image_ids.append(img_id)
                
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Trả về ảnh và ID ảnh
        return image, self.image_ids[idx]

def main():
    print(f"Sử dụng thiết bị: {DEVICE}")
    
    # Tải CLIP processor
    try:
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Lỗi khi tải CLIP processor: {e}")
        return
    
    # Lấy các thông số chuẩn hóa từ processor
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std
    image_size = processor.image_processor.size["shortest_edge"]
    
    # Định nghĩa transforms giống như trong validation
    test_transforms = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ])
    
    # Tạo dataset và dataloader
    test_dataset = MushroomTestDataset(TEST_FOLDER, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    print(f"Số lượng ảnh test: {len(test_dataset)}")
    
    # Tải mô hình CLIP
    try:
        clip_vision_model = CLIPVisionModel.from_pretrained(MODEL_NAME).to(DEVICE)
        clip_vision_model.eval()  # Đặt ở chế độ đánh giá
    except Exception as e:
        print(f"Lỗi khi tải CLIP vision model: {e}")
        return
    
    # Tạo lớp linear head với cùng kích thước như trong training
    embedding_dim = clip_vision_model.config.hidden_size
    classifier_head = nn.Linear(embedding_dim, NUM_CLASSES).to(DEVICE)
    
    # Tải trọng số đã học từ file lưu
    try:
        classifier_head.load_state_dict(torch.load(BEST_MODEL_PATH))
        classifier_head.eval()  # Đặt ở chế độ đánh giá
        print(f"Đã tải mô hình từ {BEST_MODEL_PATH}")
    except Exception as e:
        print(f"Lỗi khi tải model head: {e}")
        return
    
    # Dự đoán
    all_predictions = []
    all_image_ids = []
    
    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc="Đang dự đoán"):
            images = images.to(DEVICE)
            
            # Lấy đặc trưng từ CLIP
            image_features = clip_vision_model(pixel_values=images).pooler_output
            
            # Dự đoán lớp
            outputs = classifier_head(image_features)
            _, predicted = torch.max(outputs.data, 1)
            
            # Lưu kết quả
            all_predictions.extend(predicted.cpu().numpy())
            all_image_ids.extend(img_ids)
    
    # Lưu kết quả vào file CSV
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'type'])  # Header
        
        for img_id, pred in zip(all_image_ids, all_predictions):
            writer.writerow([img_id, pred])
    
    print(f"Đã lưu kết quả vào {OUTPUT_FILE}")
    print(f"Phân loại hoàn tất: {len(all_image_ids)} ảnh đã được phân loại")
    
    # In thống kê phân loại
    pred_counts = {}
    for pred in all_predictions:
        if pred in pred_counts:
            pred_counts[pred] += 1
        else:
            pred_counts[pred] = 1
    
    print("\nThống kê phân loại:")
    for class_idx, count in sorted(pred_counts.items()):
        print(f"Loại {class_idx} ({idx_to_class[class_idx]}): {count} ảnh")

if __name__ == '__main__':
    main() 