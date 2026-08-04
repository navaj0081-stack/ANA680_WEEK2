import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

columns = ['Sample_Code_Number', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 
           'Uniformity_of_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size',
             'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

# Load the dataset
df = pd.read_csv("Breast_Cancer_Data.csv", names=columns)

df = df.replace("?", np.nan).dropna()
df["Bare_Nuclei"] = pd.to_numeric(df["Bare_Nuclei"])

X = df.drop(columns=["Sample_Code_Number", "Class"])
y = df["Class"].map({2: 0, 4: 1})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model: Naive Bayes")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print("Model Saved as model.pkl")