from sklearn import linear_model, model_selection, metrics
import pandas as pd

dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

perceptron = linear_model.Perceptron()

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)

classificador = perceptron.fit(x_train, y_train)
Y_predito = classificador.predict(x_test)

print(metrics.accuracy_score(y_test, y_predito))

