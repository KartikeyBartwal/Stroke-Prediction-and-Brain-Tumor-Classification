# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from interpret.blackbox import LimeTabular
from interpret import show

# %% Load and preprocess data
data_loader = DataLoader()

data_loader.load_dataset()

print("Data has been loaded")

data_loader.preprocess_data()

print("Data has been preprocessed")

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()

print("Data splitting for training and testing completed")

# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)

print("Oversampling completed to balance samples for each class")

# %% Fit blackbox model
rf = RandomForestClassifier()

print("model has been loaded")

rf.fit(X_train, y_train)

print("Training finished.")

y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

print("applying lime")
# %% Apply lime
# Initilize Lime for Tabular data
lime = LimeTabular(model=rf.predict_proba, 
                   data=X_train, 
                   random_state=1)
# Get local explanations
lime_local = lime.explain_local(X_test[-20:], 
                                y_test[-20:], 
                                name='LIME')

# print(X_test[-20:])
# print(y_test[-20:])

print(type(lime_local))

show(lime_local)

print("lime has been applied")

# Keep this loop program online if you want to view those lime interpretations
while input("Press 'q' to quit: ") != 'q':
    pass

# %%
