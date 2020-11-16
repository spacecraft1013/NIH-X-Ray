import keras
import pandas
import glob
import numpy as np

X_test = np.load("data/arrays/X_test.npy")
y_test = np.load("data/arrays/y_test.npy")

data = {"Model": [],
        "Accuracy": [],
        "Loss": [],
        "MAE": []}

for file in glob.glob("data/*.h5"):
    print("Testing {}...".format(file.strip(r"data\"")))
    model = keras.models.load_model(file)
    test_loss, test_acc, test_mae = model.evaluate(X_test, y_test)
    data["Model"].append(file.strip(r"data\""))
    data["Accuracy"].append(str(test_acc*100) + "%")
    data["Loss"].append(test_loss)
    data["MAE"].append(test_mae)

df = pandas.DataFrame(data, columns=['Model', 'Accuracy', 'Loss', 'MAE'])
print(df)
df.to_csv("data/Comparison_Table.csv", index="Model")
