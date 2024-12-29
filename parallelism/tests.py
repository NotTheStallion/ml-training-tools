import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from homework import pipelined_iteration, sequential_backward, sequential_forward

def sync_parameters(model):
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)

def test_sequential_forward(full_model, local_model):
    if rank == world_size - 1:
        print("Testing sequential forward...", end="\t")
    
    inputs = torch.randn(256, 32) # inputs to the full model

    with torch.no_grad():
        dist.broadcast(inputs, src=0)  # sync data
    
    if rank == world_size - 1:
        full_output = full_model(inputs)
    else:
        full_output = None

    _, outputs = sequential_forward(local_model, inputs)

    if rank == world_size - 1:
        assert torch.allclose(full_output, outputs), f"Distributed forward output doesn't match full model output. Difference: {torch.norm(full_output - outputs)}"
        print("Passed")

def test_sequential_backward(full_model, local_model):
    if rank == world_size - 1:
        print("Testing sequential backward...", end="\t")

    inputs = torch.randn(256, 32)
    targets = torch.randn_like(inputs)
    with torch.no_grad():
        dist.broadcast(inputs, src=0)
        dist.broadcast(targets, src=0)

    if rank == world_size - 1:
        full_output = full_model(inputs)
        full_loss = nn.MSELoss()(full_output, targets)
        full_loss.backward()

    inputs, outputs = sequential_forward(local_model, inputs)
    loss = sequential_backward(inputs, outputs, targets, nn.MSELoss())

    params = {name: param.grad for name, param in local_model.named_parameters()}
    gather_list = [None for _ in range(world_size)] if rank == world_size - 1 else None
    dist.gather_object(params, gather_list, dst=world_size - 1)
    if rank == world_size - 1:
        assert torch.allclose(loss, full_loss), f"Distributed backward loss doesn't match full model loss. Difference: {torch.norm(loss - full_loss)}"

        # Merge all local named parameters into a single dictionary
        all_params = {}
        for rank_params in gather_list:
            for name, param in rank_params.items():
                all_params[name] = param

        # Compare gradients with full model
        for name, full_param in full_model.named_parameters():
            assert torch.allclose(all_params[name], full_param.grad), f"Gradient for {name} doesn't match. Difference: {torch.norm(all_params[name].grad - full_param.grad)}"
        print("Passed")

def test_pipelined_iteration(full_model, local_model):
    if rank == world_size - 1:
        print("Testing pipelined iteration...", end="\t")

    loss_fn = nn.MSELoss(reduction='sum')
    inputs = torch.randn(256, 32)
    targets = torch.randn_like(inputs)
    with torch.no_grad():
        dist.broadcast(inputs, src=0)
        dist.broadcast(targets, src=0)

    if rank == world_size - 1:
        full_output = full_model(inputs)
        full_loss = loss_fn(full_output, targets)
        full_loss.backward()

    total_loss = pipelined_iteration(local_model, inputs, targets, loss_fn)

    params = {name: param.grad for name, param in local_model.named_parameters()}
    gather_list = [None for _ in range(world_size)] if rank == world_size - 1 else None
    dist.gather_object(params, gather_list, dst=world_size - 1)

    if rank == world_size - 1:
        assert torch.allclose(torch.tensor(total_loss), full_loss), f"Distributed pipeline loss doesn't match full model loss. Difference: {abs(total_loss - full_loss.item())}"

        # Merge all local named parameters into a single dictionary
        all_params = {}
        for rank_params in gather_list:
            for name, param in rank_params.items():
                all_params[name] = param

        # Compare gradients with full model
        for name, full_param in full_model.named_parameters():
            assert torch.allclose(all_params[name], full_param.grad, atol=1e-5), f"Gradient for {name} doesn't match. Difference: {torch.norm(all_params[name] - full_param.grad)}"
        print("Passed")

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

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

    sync_parameters(model)
    full_model = copy.deepcopy(model)

    # Each rank gets a part of the model
    layers_per_rank = len(model) // world_size
    local_model = model[rank * layers_per_rank : (rank + 1) * layers_per_rank]

    test_sequential_forward(full_model, local_model)
    test_sequential_backward(full_model, local_model)
    test_pipelined_iteration(full_model, local_model)