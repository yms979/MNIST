import pandas as pd
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from skimage import  transform
import os
from PIL import Image

class MNIST(Dataset):
    def __init__(self, df, img_dir, transform=None, train=True):
        self.landmarks_frame = df
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = self.landmarks_frame.index[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')

        if self.train:
            # 학습 데이터에 대해 데이터 증강 적용
            img_transform = transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])
            ])
        else:
            # 테스트 데이터에 대해서는 증강 없이 전처리만 적용
            img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])
            ])

        image = img_transform(image)
        label = extract_label(img_name)
        sample = {'image': image, 'label': label}
        return sample

def extract_label(file_name):
    return int(file_name.split('_')[1].split('.')[0])

if __name__ == '__main__':
    img_dir = "./data/train"
    file_names = os.listdir(img_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize(mean=[0.1307], std=[0.3081])  # 이미지를 평균 0, 표준편차 1로 정규화
    ])

#   DataFrame 생성
    df = pd.DataFrame({'file_name': file_names})

    # 'label' 컬럼 추가
    df['label'] = df['file_name'].apply(extract_label)

    # 'file_name'을 index로 설정
    df.set_index('file_name', inplace=True)

    samples = MNIST(df = df, img_dir= img_dir, transform=transform)

    
