import torch

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc