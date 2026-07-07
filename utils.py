from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# get optimization method
def get_optim(model):
    return Adam(model.parameters(), 1e-3)

# get loss function
criterion = CrossEntropyLoss()