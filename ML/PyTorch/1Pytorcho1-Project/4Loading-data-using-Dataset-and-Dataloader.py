import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset
class MyDataset(Dataset):
    def __init__(self):
        # Initialize data: create a tensor of 100 points between -1 and 1, and compute their squares
        self.x = torch.linspace(-1, 1, 100).unsqueeze(1)  # 100 points between -1 and 1
        self.y = self.x.pow(2)  # Compute the square of each point

    def __len__(self):
        # Return the total number of samples
        return len(self.x)

    def __getitem__(self, idx):
        # Retrieve the sample at the given index
        return self.x[idx], self.y[idx]

# Create an instance of the custom dataset
dataset = MyDataset()

# Create a DataLoader to iterate over the dataset in batches
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Iterate over the data in batches
for batch_x, batch_y in dataloader:
    print("Batch x:", batch_x)  # Print the input batch
    print("Batch y:", batch_y)  # Print the corresponding output batch
    # Usually, you'd pass these batches to your model here.
    break  # Only print first batch for brevity