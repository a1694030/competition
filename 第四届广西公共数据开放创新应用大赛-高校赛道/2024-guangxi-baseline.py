import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 文本清洗
import re


def clean_text(text):
    # 去除标点符号和数字
    # text = re.sub(r'[^\u4e00-\u9fa5]+', '', text)
    # 去掉文本中的空格
    text = text.replace(' ', '')
    text = text.replace('...', ',')
    # text = text.replace('【','"')
    # text = text.replace('】','"')
    # text = text.replace('[','"')
    # text = text.replace(']','"')

    return text


class MultiModalDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, max_len=256, transform=None):
        try:
            self.data = pd.read_csv(csv_file)
        except Exception as e:
            self.data = pd.read_excel(csv_file)
        self.data['text'] = self.data['text'].apply(lambda x: clean_text(x))
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        if 'label' not in self.data.columns:
            self.data['label'] = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据项
        item = self.data.iloc[idx]
        text = item['text']
        label = torch.tensor(item['label'], dtype=torch.long)

        # 文本处理：tokenization + padding
        encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                  max_length=self.max_len, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        # 图像处理
        images_list = item['images_list'].split('\t') if pd.notna(item['images_list']) else []
        images = []

        for img_name in images_list:
            img_path = os.path.join(self.img_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except OSError:
                img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)

            # 应用图像变换（如果有）
            if self.transform:
                img = self.transform(img)

            images.append(img)
        # 如果有多个图像，拼接它们
        if len(images) > 1:
            images = torch.cat(images, dim=1)  # 按第二维度拼接多个图像
            # 将图像 resize 成 [3, 224, 224]
            resized_images = F.interpolate(images.unsqueeze(0), size=(224, 224),
                                           mode='bilinear', align_corners=False)
            images = resized_images.squeeze(0)
        elif len(images) == 1:
            images = images[0]
        else:
            images = torch.zeros(3, 224, 224)  # 如果没有图像，则返回零图像

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'images': images,
            'label': label
        }

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import GradScaler, autocast

class TextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TextEncoder, self).__init__()

        self.bert = BertModel.from_pretrained('/kaggle/input/bert-base-chinese/bert-base-chinese').to(device)
        # self.fc = nn.Sequential(
        #     nn.Linear(768,hidden_dim),

        # )

    def forward(self, input_ids, attention_mask):
        # cls 向量提取 bs 768
        cls = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        # cls = self.bert(input_ids,attention_mask=attention_mask).last_hidden_state.mean(dim = 1)
        # x = self.fc(cls) # bs hidden_dim
        return cls

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 224 * 224 // 16, 512)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)  # 0.3

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, text_input_dim=256, text_hidden_dim=256):
        super(MultimodalClassifier, self).__init__()
        # 文本编码器
        self.text_encoder = TextEncoder(text_input_dim, text_hidden_dim)

        # 图像编码器
        self.image_encoder = ImageEncoder()

        # 特征融合层
        self.fusion_layer = nn.Linear(768 + 512, 256)
        self.classifier = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 0.3

    def forward(self, input_ids, attention_mask, images):
        # 文本特征提取
        text_features = self.text_encoder(input_ids, attention_mask=attention_mask)

        # 图像特征提取
        image_features = self.image_encoder(images)  # [batch_size, 512]
        # 多模态特征融合
        combined_features = torch.cat((text_features, image_features),
                                      dim=1)  # [batch_size, text_hidden_dim + 512]
        fused_features = self.relu(self.fusion_layer(combined_features))
        fused_features = self.dropout(fused_features)
        # 分类层
        logits = self.classifier(fused_features)
        return logits

import gc
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer

def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

data_path = '/kaggle/input/2024-gxpublic-merge/data/train/train.csv'
model_path = '/kaggle/input/bert-base-chinese/bert-base-chinese'
image_path = '/kaggle/input/2024-gxpublic-merge/data/train/images'
test_path = './data/test.csv'
save_path = './bert_checkpoint'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 42
setup_seed(random_seed)

tokenizer = AutoTokenizer.from_pretrained(model_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset =MultiModalDataset(csv_file=data_path,img_dir=image_path,tokenizer=tokenizer,transform=transform)

# 定义训练集和评估集的长度（80% 训练，20% 评估）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
# 使用 random_split 进行划分
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
# 定义模型
model = MultimodalClassifier(num_classes=2)
# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
# 定义分类损失函数
criterion = nn.CrossEntropyLoss()
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=5,
    device=device,
    enable_visualization=True,
    is_jupyter=True
)

trainer.train()

# del tokenizer,dataset,train_dataset,train_loader,val_loader,trainer
# gc.collect()


import torch.nn.functional as F

test_path = '/kaggle/input/2024-gxpublic-merge/data/test/test.xlsx'
image_path = '/kaggle/input/2024-gxpublic-merge/data/test/images'

test_dataset =MultiModalDataset(csv_file=test_path,img_dir=image_path,tokenizer=tokenizer,transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=4)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
all_preds = []
all_proba_preds = []
with torch.no_grad():
 for batch in tqdm(test_loader):
     input_ids = batch['input_ids'].to(device)
     attention_mask = batch['attention_mask'].to(device)
     images = batch['images'].to(device)
     # 预测
     outputs = model(input_ids,attention_mask, images)
     proba_preds = F.softmax(outputs, dim=1)[:,1]
     _, preds = torch.max(outputs, dim=1)
     all_preds.extend(preds.cpu().numpy())
     all_proba_preds.extend(proba_preds.cpu().numpy())


predDF = pd.read_excel(test_path)
predDF['target'] = all_preds
predDF[['id','target']]
#保存结果
predDF[['id','target']].to_csv('predictions2.csv',index = False)