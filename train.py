import argparse
import torch
from torch import nn,optim
from torchvision import datasets, transforms, models

parser=argparse.ArgumentParser(description='Training the model')
parser.add_argument("data_dir", type = str, help="The path to our data sets")
parser.add_argument("-a","--arch", type= str, choices=["vgg13","vgg16","vgg19","alexnet"], default="vgg16", help="Picks an architecture for our model")
parser.add_argument("-s","--save_dir",type=str, help="Checkpoint directory", default="checkpoint.pth")
parser.add_argument("-lr","--learning_rate",type=float, default=0.001, help= "The model's learning rate")
parser.add_argument("-hu","--hidden_units", type = int, default=4096, choices=[4096,2048,1024,512], help="The model's hidden units")
parser.add_argument("-e","--epochs",type=int,default=3, help="The model's epochs")
parser.add_argument("-g","--gpu",action="store_true", help="If an available GPU will be used")

args=parser.parse_args()

data_dir=args.data_dir
arch=args.arch
hidden_units=args.hidden_units
learning_rate=args.learning_rate
gpu=args.gpu
epochs_num=args.epochs
save_dir=args.save_dir



if data_dir=="flowers":
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
else:
    train_dir = data_dir + str(input("Enter your training set's path:" ))
    valid_dir= data_dir + str(input("Enter your validation set's path:" ))
    test_dir = data_dir + str(input("Enter your testing set's path:"))
    
# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {"train":transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                  "valid":transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])]),
                  "test":transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])}

# TODO: Load the datasets with ImageFolder
image_datasets = {"train":datasets.ImageFolder(train_dir,transform=data_transforms["train"]),
                 "valid":datasets.ImageFolder(valid_dir,transform=data_transforms["valid"]),
                 "test":datasets.ImageFolder(test_dir,transform=data_transforms["test"])}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {"train":torch.utils.data.DataLoader(image_datasets["train"],batch_size=45,shuffle=True),
               "valid":torch.utils.data.DataLoader(image_datasets["valid"],batch_size=45),
               "test":torch.utils.data.DataLoader(image_datasets["test"],batch_size=45)}

    
model=getattr(models,arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad=False

if arch=="alexnet":
    classifier=nn.Sequential(nn.Linear(9216,hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_units,102),
                               nn.LogSoftmax(dim=1))
    print("\nYou have used", arch, "architecture for your model. Good choice!\n")
        
else:
    classifier=nn.Sequential(nn.Linear(25088,hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_units,102),
                               nn.LogSoftmax(dim=1))
    print("\nYou have used", arch, "architecture for your model. Good choice!\n")
                  

model.classifier = classifier

criterion=nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)

if gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("A GPU will be utilized if available \n")

else:
    device=torch.device('cpu')

model.to(device)

epochs=epochs_num
print_every = 10
steps=0
running_loss=0


print("Starting training... \n")

for i in range(epochs):
    for images,labels in dataloaders["train"]:
        model.train()
        steps+=1
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        logps=model(images)
        loss=criterion(logps,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
        
            with torch.no_grad():
                model.eval()
                for images,labels in dataloaders["valid"]:
                    images,labels = images.to(device),labels.to(device)
                    logps = model(images)
                    valid_loss+=criterion(logps,labels).item()
                
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            print("Epoch: {}/{}.. ".format(i+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders["train"])),
              "Valid Loss: {:.3f}.. ".format(valid_loss/len(dataloaders["valid"])),
              "Valid Accuracy: {:.3f}".format(accuracy/len(dataloaders["valid"])))
            running_loss=0

print("\nTraining is completed \n")

print("Testing model's accuracy on the testing set... \n")
with torch.no_grad():
    model.eval()
    test_loss=0
    test_accuracy=0
    for images,labels in dataloaders["test"]:
        images,labels = images.to(device),labels.to(device)
        logps = model(images)
        test_loss+=criterion(logps,labels).item()
                
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    print("Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders["test"])),
          "Test Accuracy: {:.3f}".format(test_accuracy/len(dataloaders["test"])))
                
model.class_to_idx = image_datasets['train'].class_to_idx
model.to('cpu')
                  
checkpoint = {'architecture':arch,
              'classifier':model.classifier,
              'state_dict': model.state_dict(),
              'mapping_classes': model.class_to_idx}
torch.save(checkpoint,save_dir)
          
print("Checkpoint is saved successfully")
    
