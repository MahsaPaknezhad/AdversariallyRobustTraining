import numpy as np
import torch

from copy import deepcopy

def fgsm(image, labels, model, epsilon, device, num_classes=10):
    is_cuda = device == 'cuda' and torch.cuda.is_available()
    if is_cuda:
        image = image.cuda()
        model = model.cuda()
    image_copy = deepcopy(image)
    image = torch.as_tensor(image[None, :, :, :]).requires_grad_(True)

    # Start the model in evaluation mode
    model.eval()
    # Do a forward pass of the original image through the model
    forward_pass = model(image)
    loss_func = torch.nn.CrossEntropyLoss()
    # Get the original image's label
    label_orig = torch.max(forward_pass, 1)[1]
    # label_orig = torch.tensor([labels[0]], device=torch.device('cuda'))
    loss = loss_func(forward_pass, torch.tensor([labels[0]], device=torch.device(device)))
    loss.backward(retain_graph=True)
    gradient_sign = image.grad.data.sign()

    # Create the perturbed image
    perturbed_image = image_copy + epsilon * gradient_sign
    # Get the labels of the perturbed image
    forward_pass_perturbed_img = model(perturbed_image)
    label_pert = torch.max(forward_pass_perturbed_img, 1)[1].to(device)

    # Return the perturbed image, the new label for the perturbed image, and whether the FGSM attack was successful in actually changing the label of the perturbed image.
    return perturbed_image, label_pert, int(label_orig != label_pert)
