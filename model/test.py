import torch

def test(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0

        for i, l in test_loader:
            i, l = i.to(device), l.to(device)

            outputs = model(i)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == l).sum().item()
            total += l.size(0)

        accuracy = (correct/total)*100

        print(f"Test accuracy: {accuracy:.2f}%")

