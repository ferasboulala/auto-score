from model import *
from torch.utils.data import Dataset, DataLoader


def main():
	loader = DataLoader(DatasetLoader(), batch_size=10, num_workers=10, shuffle=True)
	print(len(loader))


if __name__ == '__main__':
	main()
