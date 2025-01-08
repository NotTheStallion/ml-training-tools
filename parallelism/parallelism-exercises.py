import copy
import torch
import torch.nn as nn
import torch.distributed as dist

def column_parallel():
    x = torch.randn(4, 8)
    model = nn.Linear(8, 16, bias = False)
    column_size = 16 // world_size
    index = rank * column_size
    model.weight.data = model.weight.data[index:index + column_size, :]

    # -- Forward pass -- 
    # The input needs to be shared across all ranks
    dist.broadcast(x, src = 0)
    x = x.detach().requires_grad_()

    output = model(x)
    
    gather_list = None
    if rank == 0:
        gather_list = [torch.empty_like(output) for _ in range(world_size)]
    dist.gather(output, gather_list, dst = 0)
    if rank == 0:
        full_output = torch.cat(gather_list, dim = 1)

    # -- Backward pass --
    chunks = None
    if rank == 0:
        grad_output = torch.randn(4, 16) # assume this is the gradients
        chunks = torch.chunk(grad_output, world_size, dim = 1)
        chunks = list(chunks)
    
    grad_output = torch.empty(4, 16 // world_size)
    dist.scatter(grad_output, chunks, src = 0) # opposite of gather
    output.backward(grad_output)
    dist.reduce(x.grad.data, dst = 0, op = dist.ReduceOp.SUM) # opposite of broadcast
    
def row_parallel():
    x = torch.randn(4, 8)
    model = nn.Linear(8, 16, bias = False)
    # 1. Split the weights
    row_size = 8 // world_size
    index = rank * row_size
    model.weight.data = model.weight.data[:, index:index + row_size]

    # -- Forward pass -- 
    # 2. Split the input
    chunks = None
    if rank == 0:
        chunks = torch.chunk(x, world_size, dim = 1)
        chunks = list(chunks)
    
    x = torch.empty(4, 8 // world_size)
    dist.scatter(x, chunks, src = 0)
    x = x.detach().requires_grad_()

    output = model(x)

    # 3. Sum the output
    dist.reduce(output, dst = 0, op = dist.ReduceOp.SUM)

    # -- Backward pass --
    grad_output = torch.randn(4, 16) # assume this is the gradients
    # 4. Share the gradients
    dist.broadcast(grad_output, src = 0)

    output.backward(grad_output)

    # 5. Regroup the gradients
    gather_list = None
    if rank == 0:
        gather_list = [torch.empty_like(x.grad.data) for _ in range(world_size)]
    dist.gather(x.grad.data, gather_list, dst = 0)
    if rank == 0:
        full_grads = torch.cat(gather_list, dim = 1)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, n = 128):
        self.data = torch.randn(n, 16)
        self.targets = (self.data ** 2) - 4.2
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16, bias = False)
        self.fc2 = nn.Linear(16, 16, bias = False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
      
def tensor_parallel(model, data, criterion):
    """
    For tensor parallel training, each rank handles a part of each layer.
    """

    # Use the correct subset of the weights for each rank
    column_size = model.fc1.weight.data.shape[0] // world_size
    row_size = model.fc2.weight.data.shape[1] // world_size

    model.fc1.weight = nn.Parameter(
        model.fc1.weight.data[column_size * rank : column_size * (rank + 1), :]
    )
    model.fc2.weight = nn.Parameter(
        model.fc2.weight.data[:, row_size * rank : row_size * (rank + 1)]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(10):
        for i in range(len(data)):
            # if only one rank has the data, we need to broadcast the data here
            inputs, targets = data[i]
            optimizer.zero_grad()
            output = model(inputs)
            dist.all_reduce(output.data, op=dist.ReduceOp.SUM)
            loss = criterion(output, targets)
            # if only one rank has the targets and computes the loss, we need to broadcast the gradients here
            loss.backward()
            optimizer.step()


def data_parallel(model, data, criterion):
    """
    For data parallel training, each rank trains on a subset of the data.
    After each batch, the gradients are averaged across all ranks to maintain the same parameters.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # 1. Each rank handles a subset of the data
    # Split the dataset into equal chunks
    length = len(data) // world_size  # how much data this rank will handle
    offset = rank * length  # where to start
    local_inputs, local_targets = data[offset : offset + length]  # slice the data for this rank

    for epoch in range(10):
        for i in range(len(local_inputs)):
            # 2. Each rank trains on its own data
            optimizer.zero_grad()
            output = model(local_inputs[i])
            loss = criterion(output, local_targets[i])
            loss.backward()

            # 3. Average the gradients across all ranks
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size  # can't use ReduceOp.AVG with gloo

            # 4. Update parameters
            optimizer.step()

if __name__ == "__main__":
    dist.init_process_group(backend='gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank: {rank}, World Size: {world_size}")

    model = MyModel()
    data = MyDataset()
    criterion = nn.MSELoss()

    try:
        column_parallel()
    except Exception as e:
        print(f"Error in column parallel: {e}")

    try:
        row_parallel()
    except Exception as e:
        print(f"Error in row parallel: {e}")

    try:
        data_parallel(copy.deepcopy(model), data, criterion)
    except Exception as e:
        print(f"Error in data parallel: {e}")
    
    try:
        tensor_parallel(copy.deepcopy(model), data, criterion)
    except Exception as e:
        print(f"Error in tensor parallel: {e}")

    dist.destroy_process_group()
