import torch


class IDSModel(torch.nn.Module):
    """
    Feed forward NN for Dr. Ranwa's competition phase 0.
    Students may modify:
    - The number of layers and neurons per layer
    - The activation functions
    - Any hyperparameters such as learning rate, batch size, number of epochs, etc.
    - Any learning techniques such as regularization, dropout, etc.
    
    Students may NOT use:
    - CNNs, RNNs, Transformers, Decision Trees or any other non-FFNN (MLP) architecture
    
    Students are encouraged to experiment with:
    - Number of hidden layers
    - Neurons per layer
    - Activation functions
    - Batch normalization
    - Dropout layers
    - etc.
    """

    def __init__(self, inputSize, hiddenDims=[64, 32]):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(inputSize, hiddenDims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenDims[0], hiddenDims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenDims[1], 1),
        )

        #Binary cross entropy loss since we're doing binary classification
        self.lossFunc = torch.nn.BCEWithLogitsLoss()

        #Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    #Forward pass for the network - just pass sample sequentially through the layers
    def forward(self, sample):
        return self.network(sample)

    #Function to save the model weights (no modifications needed)
    def saveModel(self, filePath: str):
        torch.save(self.state_dict(), filePath)

    #Function to load the model weights (no modifications needed)
    def loadModel(self, filePath: str):
        self.load_state_dict(torch.load(filePath))
        self.eval()

    def fit(self, trainingData, numEpochs=1):
        """
            Train the model
            
            Students are encouraged to experiment with:

            - Optimizers (Adam, SGD, AdamW, etc.)
            - Learning rates
            - Batch sizes
            - Early stopping
            - Regularization
            - Validation monitoring
            - Learning rate schedules

            Feel free to make any modifications to this function including the arguments.
        """
        for epoch in range(0, numEpochs):
            #Implement training loop here (make any modifications you want)
            pass

        #Ensure you save the model weights (as these are what will be evaluated!)
        self.saveModel("model.pth")

    def predict(self, samples, threshold=0.5):
        """
            Generate binary predictions.

            Parameters
            ----------
            samples : array-like
                Network traffic samples.

            threshold : float
                Classification threshold.

            Returns
            -------
            np.ndarray
                Binary predictions (0=benign, 1=malicious).

            Students may experiment with:
            - Classification threshold
        """
        #Change model to evaluation mode (not strictly necessary here since we're not using dropout or batch norm, but good practice)
        self.eval()

        #Disable gradient calculation/tracking since only doing inference (more computationally efficient)
        with torch.no_grad():
            #Convert samples to tensor:
            samples = torch.tensor(samples, dtype=torch.float32).to(self.device)

            #Get the raw logits
            logits = self.forward(samples)

            #Convert the logits into probably distribution using sigmoid
            probs = torch.sigmoid(logits)

            #Convert predictions into classifications (using the specified threshold)
            predictions = (probs > threshold).cpu().numpy().astype(int).flatten()

            return predictions
