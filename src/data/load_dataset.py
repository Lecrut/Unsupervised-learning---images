from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

DATASET_NAME = "huggan/wikiart"
IMAGE_SIZE = 256


def preprocess(batch):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    batch['image'] = [transform(img.convert('RGB')) for img in batch['image']]
    return batch


def load_data(train_split=0.7, test_split=0.15, batch_size=32, num_workers=0):
    dataset = load_dataset(DATASET_NAME, split='train')
    dataset = dataset.with_transform(preprocess)
    
    train_size = int(len(dataset) * train_split)
    test_size = int(len(dataset) * test_split)
    
    splits = dataset.train_test_split(train_size=train_size, seed=42)
    train_ds = splits['train']
    temp_ds = splits['test']
    
    test_val_splits = temp_ds.train_test_split(train_size=test_size, seed=42)
    test_ds = test_val_splits['train']
    val_ds = test_val_splits['test']
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, val_loader