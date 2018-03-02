import torch
import torch.nn.functional as F
import random

class BatchWrapper:
      def __init__(self, dl, x_var, y_vars):
            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x and y
  
      def __iter__(self):
            for batch in self.dl:
                  x = getattr(batch, self.x_var) # we assume only one input in this wrapper
  
                  if self.y_vars is not None: # we will concatenate y into a single tensor
                        y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
                  else:
                        y = torch.zeros((1))
 
                  yield (x, y)
  
      def __len__(self):
            return len(self.dl)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float or division 
    acc = correct.sum()/len(correct)
    return acc

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    _, pred_cat = torch.max(preds, 1)
    correct = (pred_cat == y.long()).float()
    acc = correct.sum()/len(correct)
    return acc

def set_seeds(seed):
    """
    Sets seed for both Python and PyTorch RNGs.
    Seeds CPU, GPU and multi-GPU.
    Returns nothing
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)