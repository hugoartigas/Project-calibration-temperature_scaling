import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, loss_optim, training = True):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()
        mce_criterion = _MCELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece, before_avg_confidence_in_bin, before_accuracy_in_bin = ece_criterion(logits, labels)
        before_temperature_ece=before_temperature_ece.item()
        
        b1 = plt.bar( before_avg_confidence_in_bin, before_accuracy_in_bin, width = 0.05)
    
        print('Before temperature - NLL: %.3f, ECE/MCE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        if training :
          optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

          def eval():
              if loss_optim == "ECE" :
                loss,_,__ = ece_criterion(self.temperature_scale(logits), labels)
              if loss_optim == "MCE" :
                loss,_,__ = mce_criterion(self.temperature_scale(logits), labels)
              if loss_optim == "NLL" :
                loss= nll_criterion(self.temperature_scale(logits), labels)
              loss.backward()
              return loss
          optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece,  after_avg_confidence_in_bin, after_accuracy_in_bin = ece_criterion(self.temperature_scale(logits), labels)
        after_temperature_ece=after_temperature_ece.item()
        #print(after_avg_confidence_in_bin)
        #print(after_accuracy_in_bin)
        b2 = plt.bar( after_avg_confidence_in_bin, after_accuracy_in_bin, width = 0.05, color = 'green' )
        plt.plot(np.arange(0,1+1/10,1/10),np.arange(0,1+1/10,1/10), color='red')
        plt.xlabel("proba")
        plt.ylabel("accuracy")
        plt.legend([b1, b2], ['avant temperature_scaling', 'après temperature_scaling'])
        if training :
          plt.title("Probabilité en fonction de l'accuracy, Entrainement")
        else :
          plt.title("Probabilité en fonction de l'accuracy, Validation")
        plt.show()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE/MCE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        PROB = []
        ACCU = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                PROB +=[avg_confidence_in_bin.cpu().item()]
                ACCU +=[accuracy_in_bin.cpu().item()]
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return (ece, PROB, ACCU)



class _MCELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_MCELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        PROB = []
        ACCU = []
        MCE = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                PROB +=[avg_confidence_in_bin.cpu().item()]
                ACCU +=[accuracy_in_bin.cpu().item()]
                MCE += [torch.abs(avg_confidence_in_bin - accuracy_in_bin)]

        return (max(MCE), PROB, ACCU)


