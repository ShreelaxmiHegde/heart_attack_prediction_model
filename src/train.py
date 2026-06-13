import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

models_dir = Path("../models")
models_dir.mkdir(exist_ok=True)

df = pd.read_csv("../data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

classifier = KNeighborsClassifier()
param_grid = {"knn__n_neighbors": [3, 5, 7, 9]}

classifierCV = GridSearchCV(pipeline, param_grid, cv=5)
classifierCV.fit(X_train, y_train)

joblib.dump(
    pipeline,
    models_dir/"best_model.pkl"
)