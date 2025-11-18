#Imports 
from datasets import load_dataset

#%% Constants
DATASET_NAME = "huggan/wikiart"

#%% Download Dataset
def load_data():
    dataset_train = load_dataset(DATASET_NAME, split='train')
    dataset_test = load_dataset(DATASET_NAME, split='test')
    dataset_validation = load_dataset(DATASET_NAME, split='validation')

    print(f"Train set size: {len(dataset_train)}")
    print(f"Test set size: {len(dataset_test)}")
    print(f"Validation set size: {len(dataset_validation)}")

    return dataset_train, dataset_test, dataset_validation