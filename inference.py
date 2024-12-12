import torch
from tqdm import tqdm

def inference(model, test_loader, device, method='average'):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(test_loader):
            probs = 0
            for feature in features:
        
                prob = model(feature.to(device))
                probs += prob
            probs = probs / len(features)
            probs = probs.cpu().detach().numpy()
            #predictions.append(probs)
            predictions += probs.tolist()
            
    return predictions
                
        
        
