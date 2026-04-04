import numpy as np
from sklearn.model_selection import train_test_split
from models.deep_learning.dwe import DeepWeatherEvaluator

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
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

