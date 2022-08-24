import numpy as np 
import Loss.Parent_loss as Parent_loss




# Mean Squared Error loss
class Loss_MeanSquaredError(Parent_loss.Loss):  # L2 loss
    # Forward pass
    def forward(self, y_pred, y_true):
        
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        # Return losses
        return sample_losses
    
    
    # Backward pass
    def backward(self, d_values, y_true):

        # Number of samples
        samples = len(d_values)
        
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(d_values[0])
        
        # Gradient on values
        self.d_inputs = -2 * (y_true - d_values) / outputs
        
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples
        
        
        
        
        
# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Parent_loss.Loss):  # L1 loss
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, d_values, y_true):

        # Number of samples
        samples = len(d_values)

        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(d_values[0])

        # Calculate gradient
        self.d_inputs = np.sign(y_true - d_values) / outputs

        # Normalize gradient
        self.d_inputs = self.d_inputs / samples