import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy


class Server:
    """
    Server Class
    
    Each client as well as the global server has this class.
    """
    def __init__(self, client_idx, train_dataset, lr, epochs, batch_size, dataset_name):
        
        self.client_idx = client_idx
        
        self.train_dataset = train_dataset
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.scheduler = None
        
        self.gt = None
        self.pred = None
        
        self.log_interval = 1
        
        if self.train_dataset is not None:
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
            self.iter = iter(self.train_loader)
        else:
            self.train_loader = None
            self.iter = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.optimizer = None
        
        self.train_losses = []
        self.test_losses = []
        self.test_accuracy = []
        
        
    def train_step(self, epoch):
        """
        Runs a single training step

        Args:
            epoch (int): The current epoch we are on. 

        Returns:
            running_loss: The current loss during training
        """
        
        # for grads because only 1 epoch is possible so dataloader needs to be iterated using next
        running_loss = 0
        
        try:
            data, target = next(self.iter)
        except:
            self.iter = iter(self.train_loader)
            data, target = next(self.iter)
        
        data = data.to(self.device)
        target = target.to(self.device)
        
        if self.dataset_name == 'MNIST':
            data = torch.flatten(data, start_dim=1)
            
        self.optimizer.zero_grad()
        output = self.model(data)
        
        if self.dataset_name == 'MNIST' or self.dataset_name == 'CIFAR' or self.dataset_name == 'ImageNet' or self.dataset_name == 'CIFAR-100':
            loss = F.cross_entropy(output, target)
        elif self.dataset_name == 'Breakhis':
            loss = F.cross_entropy(output, target)
        
        loss.backward()
        self.optimizer.step()
        
        running_loss += loss.item()
        
        print('Client Idx: {} \tTrain Epoch: {} \tLoss: {:.6f}'.format(self.client_idx, epoch, loss.item()))
        
        return running_loss
    
    def train(self, val_dataset=None):
        """
        This function trains the client

        Args:
            val_dataset (dataset): The validation dataset. Defaults to None.
        """
        self.model.train()
        self.model.to(self.device)
        
        if val_dataset is not None:
            test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        
        running_loss = 0
        for e in range(1, self.epochs+1):
            loss = self.train_step(e)
            running_loss += loss
            
            if val_dataset is not None:
                self.test(test_loader) 
        
        self.train_losses.append(running_loss / self.epochs)
    
    def test(self, test_loader, isglobal=False):
        """
        This function performs testing for each client be it normal client or global server

        Args:
            test_loader (dataloader): The test dataloader
            isglobal (bool, optional): Only set for global server. Defaults to False.
        """
        self.model.eval()
        self.model.to(self.device)
        test_loss = 0
        correct = 0
        
        if isglobal:
            print('GLOBAL SERVER:')
            self.gt = []
            self.pred = []
        
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                if self.dataset_name == 'MNIST':
                    data = torch.flatten(data, start_dim=1)
            
                output = self.model(data)
                
                if self.dataset_name == 'MNIST'  or self.dataset_name == 'CIFAR' or self.dataset_name == 'ImageNet' or self.dataset_name == 'CIFAR-100':
                    test_loss += F.cross_entropy(output, target)
                    pred = F.softmax(output, dim=1).data.max(1, keepdim=True)[1]
                    
                    if isglobal:
                        self.gt.append(target.squeeze().detach().cpu().numpy())
                        self.pred.append(pred.squeeze().detach().cpu().numpy())
                
                elif self.dataset_name == 'Breakhis':
                    test_loss += F.cross_entropy(output, target)
                    pred = F.softmax(output, dim=1).data.max(1, keepdim=True)[1]
                    
                    if isglobal:
                        self.gt.append(target.squeeze().detach().cpu().numpy())
                        self.pred.append(pred.squeeze().detach().cpu().numpy())
                
                correct += pred.eq(target.data.view_as(pred)).sum()
        
        test_loss = test_loss / len(test_loader)
        self.test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 
                                                                                  100. * correct / len(test_loader.dataset)))
        
        acc = correct / len(test_loader.dataset)
        acc = acc.detach().cpu().numpy()
        self.test_accuracy.append(acc)
        
    def plot_curves(self):
        fig = plt.figure()
        
        self.test_losses = [x.cpu() for x in self.test_losses]
        
        plt.plot(self.train_losses, color='blue')
        plt.plot(self.test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('Cross Entropy Loss')
        fig.savefig(self.dataset_name + '_' + str(self.client_idx) + '_' + 'Loss.png')
        
        plt.clf()
            
        
        