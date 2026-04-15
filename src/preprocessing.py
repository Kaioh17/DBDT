# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
from hmeasure import h_score
# scikit learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
# imblearn
from imblearn.over_sampling import SMOTE
# matplot lib
import matplotlib.pyplot as plt
import torch 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
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
    
def torch_cast(X_train, X_test, y_train, y_test, device): 
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32).to(device) # .values because it's a DataFrame
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    
    return X_train_t, X_test_t, y_train_t, y_test_t
def evaluate(model, X, y):
    # get predictions and raw scores
    preds = model.predict(X).cpu().numpy()        # {-1, 1}
    
    with torch.no_grad():
        H = torch.zeros(X.shape[0]).to(X.device)
        for tree in model.trees:
            ht_out, _, _ = tree.forward(X)
            H = H + ht_out
    scores = H.cpu().numpy()                      # raw scores for AUC
    y_np = y.cpu().numpy()

    # convert {-1,1} to {0,1} for sklearn
    y_bin = ((y_np + 1) / 2).astype(int)
    p_bin = ((preds + 1) / 2).astype(int)

    # metrics
    auc    = roc_auc_score(y_bin, scores)
    f1     = f1_score(y_bin, p_bin)
    prec   = precision_score(y_bin, p_bin)
    rec    = recall_score(y_bin, p_bin)
    cm     = confusion_matrix(y_bin, p_bin)

    # h-measure
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_bin, scores)

    print(f"AUC:       {auc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")

    # confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    return {'auc': auc, 'f1': f1, 'precision': prec, 'recall': rec}
  
def evaluate_binary(y_true, y_pred, pos_label=-1):
    """
    y_true, y_pred: torch.Tensor or np.ndarray, values in {-1, +1}.
    pos_label: label treated as the positive class (here fraud = -1).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    # binary metrics need an explicit positive label when it isn't 1
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[pos_label, -pos_label])
    # auc_ = ()
    auc    = roc_auc_score(y_true, y_pred )
    
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,  # rows/cols: [pos_label, negative_label]
    }
    print(f"AUC:       {auc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    hm = h_score(y_true, y_pred)
    print(f"H-measure: {hm:.4f}")
  
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    return metrics

from collections import Counter

def print_dataset_size_levels(
    X_full, y_full,
    X_subset, y_subset,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    X_train_iqr, y_train_iqr,
    X_train_smote, y_train_smote
):
    print("=== Dataset Size Levels ===")
    print(f"Full dataset          : X={len(X_full):,}, y={len(y_full):,} | class={dict(Counter(y_full))}")
    print(f"Subset dataset        : X={len(X_subset):,}, y={len(y_subset):,} | class={dict(Counter(y_subset))}")
    print(f"Train split           : X={len(X_train):,}, y={len(y_train):,} | class={dict(Counter(y_train))}")
    print(f"Validation split      : X={len(X_val):,}, y={len(y_val):,} | class={dict(Counter(y_val))}")
    print(f"Test split            : X={len(X_test):,}, y={len(y_test):,} | class={dict(Counter(y_test))}")
    print(f"Train after IQR       : X={len(X_train_iqr):,}, y={len(y_train_iqr):,} | class={dict(Counter(y_train_iqr))}")
    print(f"Train after SMOTE     : X={len(X_train_smote):,}, y={len(y_train_smote):,} | class={dict(Counter(y_train_smote))}")

    growth = len(X_train_smote) / max(len(X_train_iqr), 1)
    print(f"\nSMOTE growth factor   : {growth:.2f}x")
