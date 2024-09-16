import torch
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




# Set the hyperparameters
NUM_CLASSES = 4
NUM_FEATURES = 3  # Only the mean is used as a feature



device = "cuda" if torch.cuda.is_available() else "cpu"
# Generate the dataset
y_blob = np.array([0]*100 + [1]*100 + [2]*100 + [3]*100)
array = np.zeros((400, 50))
i = 0
j = 0
z = 0
with open('output.txt', 'r') as file:
    for line in file:
        if z != 35:  # Skip lines until z reaches 35
            z = z + 1
            continue
        z = 0  # Reset z 
        if line[0] == '0':
            line = line[1:].strip()  # Remove the first character if it's '0'
        number = int(line)
        array[i][j] = float(number)
        j = j + 1
        if j == 50:  
            i = i + 1
            j = 0
            if i == 100:
                break


            
with open('output2.txt', 'r') as file:
   for line in file:
        if z != 35:  
            z = z + 1
            continue
        z = 0  
        if line[0] == '0':
            line = line[1:].strip()  
        number = int(line)
        array[i][j] = float(number)  
        j = j + 1
        if j == 50: 
            i = i + 1
            j = 0
            if i == 200:
                break





j = 0
z = 0
with open('output3.txt', 'r') as file:
    for line in file:
            if z != 35:  
                z = z + 1
                continue
            z = 0  
            if line[0] == '0':
                line = line[1:].strip()  
            number = int(line)
            array[i][j] = float(number)  
            j = j + 1
            if j == 50:  
                i = i + 1
                j = 0
                if i == 300:
                    break

j = 0
z = 0
with open('outputstay.txt', 'r') as file:
    for line in file:
        if line[0] == '0':
            line = line[1:].strip()  
        number = int(line)
        array[i][j] = float(number)  
        j = j + 1
        if j == 50:  
            i = i + 1
            j = 0
            if i == 400:
                break
print(array)

meanarr = np.zeros(400)
stdarr = np.zeros(400)
numoutlier = np.zeros(400)
DxDtchange = np.zeros(400)
i = 0
for each in array:
    meanarr[i]=np.mean(each)
    filtered = each
    stdarr[i]=np.std(filtered)
    filtered_arr = each[each<= 100]
    abs_diff = np.abs(np.diff(filtered_arr))
    result = np.sum(abs_diff)
    DxDtchange[i] = result
    num_out = 0
    for j in range(50):
        if(each[j]>50):
            num_out = num_out +1
    numoutlier[i] = num_out
    i = i + 1

X_blob = np.column_stack((stdarr, numoutlier, DxDtchange))

# Convert to tensors
X_blob = torch.from_numpy(X_blob).type(torch.float32)
y_blob = torch.from_numpy(y_blob).type(torch.long)


# Split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
   X_blob, y_blob, test_size=0.2
)

#the model
class Model(nn.Module):
   def __init__(self, input_features, output_features, hidden_units=16):
       super().__init__()
       self.linear_layer_stack = nn.Sequential(
           nn.Linear(input_features, hidden_units),
           nn.ReLU(),
           nn.Linear(hidden_units, hidden_units),
           nn.ReLU(),
           nn.Linear(hidden_units, hidden_units),
           nn.ReLU(),
           nn.Linear(hidden_units, output_features)
       )
 
   def forward(self, x):
       return self.linear_layer_stack(x)

model = Model(input_features=NUM_FEATURES, output_features=NUM_CLASSES, hidden_units=16).to(device)

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#accuracy function
def accuracy_fn(y_true, y_pred):
   correct = (y_true == y_pred).sum().item()
   return correct / len(y_true) * 100

# Training loop
epochs = 1000
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model.train()
    y_logits = model(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_fn(y_logits, y_blob_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_logits = model(X_blob_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_loss = loss_fn(test_logits, y_blob_test)
            test_acc = accuracy_fn(y_blob_test, test_pred)
            print(f"Test Accuracy: {test_acc:.2f}")




# Test Accuracy
model.eval()
with torch.no_grad():
   test_logits = model(X_blob_test)
   test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
   test_acc = accuracy_fn(y_blob_test, test_pred)
   print(f"Final Test Accuracy: {test_acc:.2f}")




array = np.zeros(50)


j = 0

with open('outputstay2.txt', 'r') as file:
 for line in file:
        if z != 35:  # Skip lines until z reaches 35
            z = z + 1
            continue
        z = 0  # Reset z after every 35th line
        if line[0] == '0':
            line = line[1:].strip()  # Remove the first character if it's '0'
        number = int(line)
        array[j] = float(number)  # Store the number in the array
        j = j + 1
        if j == 50:  # If 50 numbers are stored, reset j and increment i
            break
print(array)

res = np.zeros(3)


res[0] = np.std(array)
filtered_arr = array[array<= 100]
abs_diff = np.abs(np.diff(filtered_arr))
res[2] = np.sum(abs_diff)
for j in range(50):
   if(array[j]>50):
       num_out = num_out +1
res[1] = num_out

print("yes")
print(res)



testsample = torch.from_numpy(res).type(torch.float32).to(device)





testsample = testsample.unsqueeze(0)  # Add a batch dimension


test_logits = model(testsample)
test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
print(test_pred)
torch.save(model.state_dict(), "modelF.pth")
