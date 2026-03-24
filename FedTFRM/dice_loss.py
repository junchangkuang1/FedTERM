import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        # print(self.inter, self.union, t,torch.sum(input),torch.sum(target))
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).zero_().to(input.device)
    else:
        s = torch.FloatTensor(1).zero_()

    length = 0
    #
    # intersection = torch.sum(input * target)
    # dice = (2. * intersection+0.00001) / (torch.sum(input) + torch.sum(target)+0.00001)
    # return dice
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
        length = i
    return s / (length + 1)
