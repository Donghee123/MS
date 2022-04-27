import torch
def test(model, dev_samples, nBatchSize):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_y_list = []
    pred_y_list = []

    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dev_samples, batch_size=nBatchSize, shuffle=True)

        for data, true_y in test_loader:
            data = data.float().to(device)
            true_y = true_y.to(device)           
                
            output = model(data)
            pred_y = torch.argmax(output, axis=1)

            pred_y_list.extend(pred_y.tolist())
            true_y_list.extend(true_y.tolist())

    train_accuracy =  accuracy_score(true_y_list, pred_y_list)
    return train_accuracy

def accuracy_score(label_y, predict_y):
    
    correct_count = 0
    for index, value in enumerate(label_y):
        if predict_y[index] == value:
            correct_count += 1
    
    return correct_count / len(label_y)