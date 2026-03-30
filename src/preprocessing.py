# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
# Set the path to the file you'd like to load
file_path = "creditcard.csv"

df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "mlg-ulb/creditcardfraud",
  file_path,

  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

def load_data(df):
    """Separate features and labels.
    return: X, y"""
    y = df['Class'] 
    y = np.where(y == 1, -1, 1) # Convert labels as the paper uses [-1, 1] (3.1.2)
    X = df.drop(columns=['Class'])

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print(np.unique(y))
    return X, y
    
load_data(df)

