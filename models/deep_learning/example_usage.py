import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch
from dwe import DeepWeatherEvaluator

def generate_synthetic_data(num_samples=1000, num_features=5):
    """
    Generate synthetic weather data for training a deep learning model.
    """
    np.random.seed(42)
    # example porperties:
    # property_names = ["temperature", "humidity", "wind_speed", "pressure", "cloud_cover"]
    # label_name = "rain_tomorrow"

    # Generate binary labels for the last column (0 or 1)
    labels = np.random.randint(0, 2, size=(num_samples, 1))
    features = np.random.rand(num_samples, num_features)  # Random float values for the properties

    # Combine features and labels
    data = np.hstack([features, labels])

    print("Synthetic data shape:", data.shape)
    print("First 5 rows:")
    print(data[:5])
    print("\nLast column value counts:")
    unique, counts = np.unique(data[:, -1], return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  {val}: {count} samples")

    # Example usage: accessing the data
    # X = data[:, :-1]  # Features (first 5 columns)
    # y = data[:, -1]   # Labels (last column)

    return data

if __name__ == "__main__":
    data = generate_synthetic_data()

    # split dataset into features and labels
    X = data[:, :-1]  # Features (first 5 columns)
    y = data[:, -1]   # Labels (last column)

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)  # add an extra dimension for the labels
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
    y_test_tensor = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1)

    print("Training tensor set shape:", X_train_tensor.shape, y_train_tensor.shape)
    print("Testing tensor set shape:", X_test_tensor.shape, y_test_tensor.shape)

    # initialize the model
    model = DeepWeatherEvaluator(in_dimension=X_train_tensor.shape[1], out_dimension=1)
    
    # train the model
    model.fit(X_train_tensor, y_train_tensor, learning_rate=0.01, epochs=1000)
    
    # evaluate the model on the test set
    y_pred = model.predict(X_test_tensor)
    accuracy = (y_pred == y_test_tensor).float().mean()

    print(f"Test accuracy: {accuracy.item():.6f}")
