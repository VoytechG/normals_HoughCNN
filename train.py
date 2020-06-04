import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import pickle
import os

torch.backends.cudnn.benchmark = True

from config import *


# create the model
print("creating the model")
if scale_number == 1:
    import models.model_1s as model_def
elif scale_number == 3:
    import models.model_3s as model_def
elif scale_number == 5:
    import models.model_5s as model_def
net = model_def.create_model()

# load the dataset
print("loading the model")
dataset = pickle.load(open(os.path.join(dataset_directory, dataset_filename), "rb"))

print("Creating optimizer")
criterion = nn.MSELoss()
optimizer = optim.SGD(
    net.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9
)

if USE_CUDA:
    net.cuda()
    criterion.cuda()

dataset["input"] -= dataset["mean"][None, :, :, :]

input_data = torch.from_numpy(dataset["input"]).float()
target_data = torch.from_numpy(dataset["targets"]).float()
ds = torch.utils.data.TensorDataset(input_data, target_data)
ds_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

print("Training")

if not os.path.exists(result_directory):
    os.makedirs(result_directory)

np.savez(os.path.join(result_directory, "mean"), dataset["mean"])
f = open(os.path.join(result_directory, "logs.txt"), "w")

for epoch in range(epoch_max):

    if epoch % decrease_step == 0 and epoch > 0:
        learning_rate *= drop_learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    total_loss = 0
    count = 0

    t = tqdm(ds_loader, ncols=80)
    for data in t:

        # set optimizer gradients to zero
        optimizer.zero_grad()

        # create variables
        batch = Variable(data[0])
        batch_target = Variable(data[1])
        if USE_CUDA:
            batch = batch.cuda()
            batch_target = batch_target.cuda()

        # forward backward
        output = net.forward(batch)
        error = criterion(output, batch_target)
        error.backward()
        optimizer.step()

        count += batch.size(0)
        total_loss += error.item()

        t.set_postfix(Bloss=error.item() / batch.size(0), loss=total_loss / count)

    f.write(str(epoch) + " ")
    f.write(str(learning_rate) + " ")
    f.write(str(total_loss))
    f.write("\n")
    f.flush()

    # save the model
    torch.save(net.state_dict(), os.path.join(result_directory, "state_dict.pth"))

f.close()
