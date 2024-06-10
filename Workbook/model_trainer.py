from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tqdm


class ModelTrainer:
    def __init__(self, net, train_dataloader, loss_criterion=nn.BCELoss(), learning_rate=0.0001, num_epochs=5, num_classes=2):
        self.net = net
        self.train_dataloader = train_dataloader
        self.loss_criterion = loss_criterion
        self.optimizer = optim.Adam(net.parameters(), learning_rate)
        self.num_epochs = num_epochs
        self.device = torch.device("mps" ) #if torch.backends.mps.is_available() else "cpu"
        self.classes = num_classes


    def train(self, val_dataloader=None):
        self.net.to(self.device)
        self.net.train()
        
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times

            start_time = time.time()
             # Wrap train_dataloader with tqdm for progress bar
            loss_epoch = 0
            for i, data in enumerate(self.train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                labels = torch.squeeze(labels, dim=1)
                # Dimension überprüfen
                # print(labels.shape)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)

                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_epoch = loss_epoch + loss
                
            end_time = time.time()
            print(f"epoch {epoch + 1} took {end_time - start_time} seconds.")
            print(f"Average train loss during epoch: {loss_epoch/i}")
            if self.classes > 2:
                self.multiclass_test(self.train_dataloader, show_plot=False, eval_type="Train")
                if val_dataloader:
                    self.multiclass_test(val_dataloader, show_plot=False)
            else:
                self.binary_test(self.train_dataloader, show_plot=False, eval_type="Train")
                if val_dataloader:
                    self.binary_test(val_dataloader, show_plot=False)
            print("")

        print('Finished Training')

    def multiclass_test(self, eval_dataloader, show_plot=True, eval_type="Validation"):
        """
        Based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        """
        self.net.eval()

        class_correct = list(0. for i in range(self.classes))
        class_total = list(0. for i in range(self.classes))

        overall_correct = 0
        overall_total = 0
        loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for j, data in enumerate(eval_dataloader):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                labels = torch.squeeze(labels, dim=1)
                outputs = self.net(images)
                
                i=+1
                loss += self.loss_criterion(outputs, labels).item()
                outputs = torch.softmax(outputs, dim=1)

                _, predicted = torch.max(outputs, 1) # equivalent to argmax

                all_preds.extend(predicted.cpu())
                all_labels.extend(labels.cpu())

                overall_total += labels.size(0)
                overall_correct += (predicted == labels).sum().item()
                correct = (predicted == labels).squeeze()
                
                for i, label in enumerate(labels):
                    
                    class_correct[label] += correct[i].item() if correct.dim() > 0 else correct.item() # if wurde aufgrund von Fehlermeldung ergänzt: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
                    class_total[label] += 1

        overall_accuracy = overall_correct / overall_total

        print(f'{eval_type} loss: {loss/j}')
        print(f'{eval_type} accuracy: {overall_accuracy}')
            

        if show_plot:
            for i in range(self.classes):
                print(f'{eval_type} accuracy of %5s : %2d %%' % (
                    i, 100 * class_correct[i] / class_total[i]))

            conf_matrix = confusion_matrix(all_labels, all_preds )
            display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
            display.plot()
            plt.show()

    def binary_test(self, eval_dataloader, show_plot=True, eval_type="Validation"):
        self.net.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in eval_dataloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                outputs = torch.round(outputs)
                
                all_preds.extend(outputs.cpu())
                all_labels.extend(labels.cpu())

        
        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        print(f"Overall {eval_type} Accuracy: {accuracy}")
        display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        if show_plot == True:
            display.plot()
            plt.show()