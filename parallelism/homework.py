import torch
import torch.nn as nn
import torch.distributed as dist

"""
The goal of this homework is to implement a pipelined training loop.
Each rank holds one part of the model. It receives inputs from the previous rank and sends outputs to the next rank.

GPipe schedule:
--------------------------------------
Rank 0 | F F F F             B B B B |
Rank 1 |   F F F F         B B B B   |
Rank 2 |     F F F F     B B B B     |
Rank 3 |       F F F F B B B B       |
--------------------------------------

Command to run this file:
torchrun --nproc-per-node 4 homework.py
"""


def sequential_forward(model_part, inputs):
    """
    Handles the forward pass in a distributed pipeline
    
    - For all ranks except the first (rank 0), receives inputs from the previous rank
    - Processes the inputs through the local model segment
    - For all ranks except the last, sends the outputs to the next rank
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank != 0:
        # Receive inputs from the previous rank
        prev_rank = rank - 1
        inputs = torch.zeros_like(inputs)
        try:
            dist.recv(inputs, src=prev_rank)
        except Exception as e:
            print(f"[Rank {rank}] Seq_f Error receiving inputs from rank {prev_rank}: {e}")
            raise

    inputs.requires_grad_()

    try:
        outputs = model_part(inputs)
    except Exception as e:
        print(f"[Rank {rank}] Seq_f Error during forward pass: {e}")
        raise

    if rank != world_size - 1:
        # Send outputs to the next rank
        next_rank = rank + 1
        try:
            dist.send(outputs, dst=next_rank)
        except Exception as e:
            print(f"[Rank {rank}] Seq_f Error sending outputs to rank {next_rank}: {e}")
            raise

    return inputs, outputs

def sequential_backward(inputs, outputs, targets, loss_fn):
    """
    Executes a backward pass in a pipeline-parallel distributed setup
    
    - Last rank computes the loss and backwards from there
    - Other ranks receive gradients from the next rank and perform backward on outputs with received gradients
    - All ranks except first send gradients to the previous rank

    hint: tensor.backward() can take a gradient tensor as an argument
    
    Returns the loss on the last rank
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == world_size - 1:
        # Compute loss and backward
        try:
            loss = loss_fn(outputs, targets)
            loss.backward()
        except Exception as e:
            print(f"[Rank {rank}]  Seq_b Error during loss computation or backward pass: {e}")
            raise
    else:
        # Receive gradients from the next rank and backward
        next_rank = rank + 1
        grad_outputs = torch.zeros_like(outputs)
        try:
            dist.recv(grad_outputs, src=next_rank)
            outputs.backward(grad_outputs)
        except Exception as e:
            print(f"[Rank {rank}] Seq_b Error receiving gradients from rank {next_rank} or during backward pass: {e}")
            raise

    if rank != 0:
        # Send gradients to the previous rank
        prev_rank = rank - 1
        grad_outputs = outputs.grad
        try:
            dist.send(grad_outputs, dst=prev_rank)
        except Exception as e:
            print(f"[Rank {rank}]  Seq_b Error sending gradients to rank {prev_rank}: {e}")
            raise

    if rank == world_size - 1:
        return loss

def pipelined_iteration(model_part, inputs, targets, loss_fn):
    """
    Implement one iteration of pipelined training using GPipe
    - Split the inputs and targets into microbatches
    - Perform forward passes for all microbatches (use sequential_forward)
    - Perform backward passes for all microbatches (use sequential_backward)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    microbatches = torch.chunk(inputs, world_size)
    microtargets = torch.chunk(targets, world_size)
    
    total_loss = 0
    forward_outputs = []

    # Forward pass for all microbatches
    for i, microbatch in enumerate(microbatches):
        _, outputs = sequential_forward(model_part, microbatch)
        forward_outputs.append(outputs)

    # Backward pass for all microbatches
    for i, (microbatch, microtarget) in enumerate(zip(microbatches, microtargets)):
        microbatch.requires_grad_()
        print("="*10,microbatch)
        loss = sequential_backward(microbatch, forward_outputs[i], microtarget, loss_fn)
        if rank == world_size - 1:
            total_loss += loss.item()

    return total_loss

class MyDataset(torch.utils.data.Dataset):
    """
    Dummy dataset for testing
    """
    def __init__(self, n = 1024):
        self.data = torch.randn(n, 32)
        self.targets = (self.data * 1.3) - 0.65
        # Synchronize data across all ranks
        with torch.no_grad():
            dist.broadcast(self.data, src = 0)
            dist.broadcast(self.targets, src = 0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def pipelined_training(model_part):
    """
    Perform pipelined training on a full dataset
    For each batch:
    - Perform pipelined iteration (use pipelined_iteration)
    - Update the model parameters
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dataset = MyDataset()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model_part.parameters())
    batch_size = 8
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(10):
        epoch_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            loss = pipelined_iteration(model_part, inputs, targets, loss_fn)
            optimizer.step()
            if rank == world_size - 1:
                epoch_loss += loss

        if rank == world_size - 1:
            print(f"[Rank {rank}] Epoch {epoch} loss: {epoch_loss / len(data_loader)}")

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # This is the full model
    model = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.Identity() # an even number of layers is easier to split
    )

    # Each rank gets a part of the model
    layers_per_rank = len(model) // world_size
    local_model = model[rank * layers_per_rank : (rank + 1) * layers_per_rank]
    print(f"Rank {rank} model: {local_model}")

    inputs = torch.randn(256, 32) # inputs to the full model
    targets = torch.randn(256, 32) # targets

    try:
        inputs, outputs = sequential_forward(local_model, inputs)
        print(f"[Rank {rank}] Sequential forward succeeded")
    except Exception as e:
        print(f"[Rank {rank}] Sequential forward failed with error: {e}")

    try:
        sequential_backward(inputs, outputs, targets, nn.functional.mse_loss)
        print(f"[Rank {rank}] Sequential backward succeeded")
    except Exception as e:
        print(f"[Rank {rank}] Sequential backward failed with error: {e}")

    try:
        pipelined_iteration(local_model, inputs, targets, nn.functional.mse_loss)
        print(f"[Rank {rank}] Pipeline iteration succeeded")
    except Exception as e:
        print(f"[Rank {rank}] Pipeline iteration failed with error: {e}")

    try:
        pipelined_training(local_model)
        print(f"[Rank {rank}] Pipeline training succeeded")
    except Exception as e:
        print(f"[Rank {rank}] Pipeline training failed with error: {e}")

    dist.destroy_process_group()

"""
Additional question (optional):

Megatron-LM (https://arxiv.org/pdf/2104.04473) proposes a mechanism called "interleaving" (Section 2.2). Its idea is to assign multiple stages to each rank, instead of one.
- What is the main benefit of this approach?
- What is the main drawback?
- What would you change in the code to implement this?
"""