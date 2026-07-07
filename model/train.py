import torch

def train(num_epochs: int, model, train_loader, device, optim, criterion):
    for epoch in range(num_epochs):
        training_loss = 0

        model.train()

        for i, l in train_loader:
            i, l = i.to(device), l.to(device)

            optim.zero_grad()

            outputs = model(i)
            loss = criterion(outputs, l)
            loss.backward()

            optim.step()

            training_loss += loss.item()

        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {training_loss/len(train_loader):.3f}")
        torch.save(model.state_dict(), "intel_model.pth")
