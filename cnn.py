import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms 
import torch.optim as optim
import wandb

# load data (cifar10)
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

batch_size = 4 

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform,
)

trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)


# define model 
class NN(nn.Module): 

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,3,kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(3,1,kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(1,1, kernel_size=1, stride=1)
        self.fc = nn.Linear(784,10)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x,1)
        print(x.shape)
        x = self.fc(x)
        # print(x.shape)

        return x
    


model = NN()

# inputs = torch.rand(1,1,12,12)
# out = model(inputs)
# print(out)


# define optimizer, loss function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs=10

# initialize wandb 
wandb.init(
    project="graph-neural-operator",
    name="simple-test",
    config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "lr": 0.001,
        "momentum": 0.9
    }
)
config = wandb.config 

# loop epochs
for epoch in range(1,num_epochs+1):

    epoch_loss = 0.0

    # loop batches 
    num_batches = len(trainloader)
    batch_loss = 0.0
    for i, data in enumerate(trainloader):

        inputs, labels = data

        optimizer.zero_grad()

        # forward 
        outputs = model(inputs)
        
        # compute loss 
        loss = loss_fn(outputs, labels)

        # backprop 
        loss.backward()

        # update gradients 
        optimizer.step()

        # update loss 
        batch_loss += loss.item()
    
    # epoch loss 
    epoch_loss = (batch_loss / num_batches)
    wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})
    print(f"epoch {epoch} loss: {epoch_loss}")

wandb.finish()

    

# define testing loop 




# weights and biases? 