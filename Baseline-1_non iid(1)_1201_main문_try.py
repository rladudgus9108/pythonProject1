import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.optim import lr_scheduler
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, mnist_dataset, label_nums):
        self.data = []
        self.redata = []

        # label_num에 속하는 label 별로 리스트 생성 후 데이터 모음
        label_data = {label: [] for label in label_nums}

        for image, label in mnist_dataset:
            if label in label_nums:
                label_data[label].append((image, label))
            else:
                self.redata.append((image, label))

        for num in label_nums:
            ten_len = int(len(label_data[num]) * 0.1)

            self.data.extend(label_data[num][:ten_len])
            self.redata.extend(label_data[num][ten_len:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = image.to(device)
        return image, label

    def get_redata(self):
        return self.redata


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(3, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def client_update(client_model, optimizer, train_loader, redata_loader, epoch, criterion):
    test(client_model, redata_loader, criterion)

    client_model.to(device)
    client_model.train()

    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return loss.item()


def server_aggregate(global_model, client_models):
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)

    global_model.load_state_dict(global_dict)

    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def test(global_model, test_loader, criterion):
    test_loss = 0
    correct = 0

    global_model.to(device)
    global_model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = global_model(data)
            outputs_softmax = F.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs_softmax, dim=1)
            test_loss += criterion(outputs, target).item()
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc


def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values: \n{param.data}")


if __name__ == '__main__':

    print_acc = 0
    print_round = 0

    num_clients = 10
    num_rounds = 200
    num_local_epochs = 1
    batch_size = 100
    learning_rate = 0.01
    seed = 2  # seed를 고정해서 실험을 돌려야 함

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),
                                              download=True)

    random.seed(seed)  # python 무작위 연산 random seed 고정
    np.random.seed(seed)  # numpy 무작위 연산 random seed 고정
    torch.manual_seed(seed)  # pytorch 무작위 연산 random seed 고정

    if torch.cuda.is_available():
        # torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    total_samples = train_dataset.data.shape[0]
    samples_per_client = total_samples // num_clients  # 정수 나누기를 사용
    remainder = total_samples % num_clients  # 나머지 연산

    # 각 클라이언트에 할당될 데이터 샘플 수를 리스트로 구성
    lengths = [samples_per_client + 1 if i < remainder else samples_per_client for i in range(num_clients)]

    traindata_split = torch.utils.data.random_split(train_dataset, lengths)

    random_list = []

    perA = np.random.permutation(10)
    perB = np.random.permutation(10)
    perC = np.concatenate((perA, perB), axis=0)
    random_list = perC.reshape(-1, 2)

    traindata_split_preprocessing = [CustomImageDataset(traindata_split[i], random_list[i]) for i in range(num_clients)]

    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in
                    traindata_split_preprocessing]
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    redata_loader = [torch.utils.data.DataLoader(x.redata, batch_size=batch_size, shuffle=False) for x in
                     traindata_split_preprocessing]

    global_model = CNNMnist().to(device)
    global_model.eval()  # 확실하게 하고자 설정, 동일한 값이 나오는 것 확인완료
    client_models = [CNNMnist().to(device) for _ in range(num_clients)]

    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    criterion_test = nn.CrossEntropyLoss()
    criterions = [nn.CrossEntropyLoss() for _ in range(num_clients)]
    optimizers = [torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5) for model in client_models]

    schedulers = [lr_scheduler.ExponentialLR(optimizier, gamma=0.995) for optimizier in optimizers]

    test_accuracy_list = []
    specific_store_list = []
    client_idx = [x for x in range(num_clients)]

    for round in range(1, num_rounds + 1):
        # client update
        loss = 0
        for i in range(num_clients):
            loss += client_update(client_models[i], optimizers[i], train_loader[i], redata_loader[i], num_local_epochs,
                                  criterions[i])

        loss /= num_clients
        # server aggregate
        server_aggregate(global_model, client_models)

        # In paper, lr is initialized as 0.01 and exponentially decayed by 0.995 over communication rounds on the MNIST dataset
        for scheduler in schedulers:
            scheduler.step()
        print("------------------------Accuracy in global model------------------------")
        # validataion
        test_loss, acc = test(global_model, test_loader, criterion_test)

        test_accuracy_list.append(acc)
        if acc >= print_acc:
            print_acc = acc
            print_round = round

        if round == 200 or round == 300 or round == 500:
            values = [round, print_round, print_acc]
            specific_store_list.append(values)
        print(
            "Round : {}, Local_train_loss : {:.5f}, Global_validation_loss : {:.5f}, Global_validation_Accuracy: {:.5f}\n"
            "Best_G_validation_Accuracy : {:.5f}, Best_Round : {}".format(round, loss, test_loss, acc,
                                                                          print_acc, print_round))

    print(seed, specific_store_list)
