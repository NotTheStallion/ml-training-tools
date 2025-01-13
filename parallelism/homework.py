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
        inputs = torch.zeros_like(inputs, requires_grad=True)
        try:
            dist.recv(inputs, src=prev_rank)
        except Exception as e:
            print(f"[Rank {rank}] Seq_f Error receiving inputs from rank {prev_rank}: {e}")
            raise

    try:
        outputs = model_part(inputs)
        outputs.retain_grad()
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
            outputs.retain_grad()
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
            outputs.retain_grad()
            outputs.backward(grad_outputs)
        except Exception as e:
            print(f"[Rank {rank}] Seq_b Error receiving gradients from rank {next_rank} or during backward pass: {e}")
            raise

    if rank != 0:
        # Send gradients to the previous rank
        prev_rank = rank - 1
        
        grad_outputs = outputs.grad
        try:
            dist.send(inputs.grad, dst=prev_rank)
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
    forward_inputs = []
    forward_outputs = []

    # Forward pass for all microbatches
    for i, microbatch in enumerate(microbatches):
        inputs, outputs = sequential_forward(model_part, microbatch)
        forward_inputs.append(inputs)
        forward_outputs.append(outputs)

    # Backward pass for all microbatches
    for i, (microbatch, microtarget) in enumerate(zip(microbatches, microtargets)):
        loss = sequential_backward(forward_inputs[i], forward_outputs[i], microtarget, loss_fn)
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

# * Answers :
# - The main benifit of using interleaving as introduced by Megatron-LM is that it allows for better utilization of the GPUs. Without interleaving, the GPUs would be either idle for a significant amount of time or computes heavy computations (the backward is usually heavier than the forward pass) due to the nature of computations asked from it. Interleaving allows for the GPUs to be utilized more efficiently by allowing them to perform different tasks at the same time and not only one. The images (https://developer-blogs.nvidia.com/wp-content/uploads/2021/03/interleaved_1F1B_schedule-1-625x288.png) shows a clear example of how interleaving can help in better utilization of the GPUs.

# - The principal drawback of interleaving is that it requires more GPU from each individual GPU as well as more communications between GPUs due to the fact that each GPU is now responsible for another task that uses gradients from another GPU. This can lead to a communication bottleneck and can slow down the training process.

# - The two main changes that would need to be made in the code to implement interleaving are:
#     1. Changing the model partitioning method to chunks of the model to each rank
#     2. Modify the forward and backward passes to handle multiple stages.
#     3. Synchronize the gradients between the different stages of the model.
#
#  The changes are talked about in the NVIDIA blog post (https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)