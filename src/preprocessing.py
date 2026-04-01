# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
# scikit learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
# imblearn
from imblearn.over_sampling import SMOTE
# matplot lib
import matplotlib.pyplot as plt

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

def load_data():
    """Separate features and labels.
    return: X, y"""
    
    y = df['Class'] 
    y = np.where(y == 1, -1, 1) # Convert labels as the paper uses [-1, 1] (3.1.2)
    X = df.drop(columns=['Class'])

    return X, y
     
def test_train_split(X, y, test_size=0.25, random_state=42):
    """Z-score Standerdization on traininig features [Amount,Time]
       
      return: X_train, X_test, y_train, y_test"""
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state) # stratify argument to maintain similar data distribution due to imbalance
    
    X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])  
    X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])  
    
    return X_train, X_test, y_train, y_test
def train_valid_split(X_train, y_train, test_size=0.25, random_state=42):
    """Validation split form training
       
      return: X_train, X_valid, y_train, y_valid"""

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train, random_state=random_state) # stratify argument to maintain similar data distribution due to imbalance
     
    return X_train, X_valid, y_train, y_valid
def interquatile_range(X_train, y_train):
  """ 
  IQR outlier detection and removal on training set only 
  Detects outliers on Amount and Time Features only. since V1-V28 are already clean PCA outputs and aggressive filtering there could accidentally remove genuine fraud cases.
  return: X_train_clean, y_train_clean 
  """  
  cols = ['Amount','Time']

  q1 = X_train[cols].quantile(0.25)
  q3 = X_train[cols].quantile(0.75) 
  
  IQR = q3 - q1
  
  
  lower_fence = q1 - 1.5 * IQR
  upper_fence = q3 + 1.5 * IQR
  
  mask = ((X_train[cols] >= lower_fence) & (X_train[cols] <= upper_fence)).all(axis=1)
  X_train_clean = X_train[mask]
  y_train_clean = y_train[mask]
  return X_train_clean, y_train_clean
# Apply SMOTE to training set only — do not touch test set
def apply_smote(X_train_clean, y_train_clean, random_state=42):
  """
  APPLY SMOTE AFTER IQR. Hence the `*_train_clean`
  For high imbalance in training set. (0.17% Fraud cases)
  
  SMOTE fixes this by creating syntethic fraud samples.
  using imb learn import `from imblearn.over_sampling import SMOTE`
  """
  
  smote = SMOTE(random_state=random_state)
  
  X_train_resampled, y_train_resampled = smote.fit_resample(X_train_clean, y_train_clean)
  
  return X_train_resampled, y_train_resampled

def plot_tsne(X, y, title="t-SNE Visualization", sample_size=5000, random_state=42):
    """t-SNE visualization of class separation """
    # Sample subset for computational efficiency
    idx = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values
    y_sample = y[idx] if isinstance(y, np.ndarray) else y[idx]
    
    # Fit and transform
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    X_embedded = tsne.fit_transform(X_sample)
    
    # Plot
    fraud_mask = y_sample == -1
    legit_mask = y_sample == 1
    
    plt.figure(figsize=(10, 7))
    plt.scatter(X_embedded[legit_mask, 0], X_embedded[legit_mask, 1], 
                c='steelblue', label='Legit', alpha=0.4, s=10)
    plt.scatter(X_embedded[fraud_mask, 0], X_embedded[fraud_mask, 1], 
                c='crimson', label='Fraud', alpha=0.7, s=15)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()