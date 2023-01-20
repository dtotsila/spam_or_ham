
from sklearn.neighbors import KNeighborsClassifier
from utils import read_data, plot_cf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


# Read Train Test data
X_train, y_train = read_data('train')
X_test, y_test = read_data('test')

# init classifier n chosen
knc = KNeighborsClassifier(n_neighbors=200)

# train classifier
knc.fit(X_train, y_train)

# test
y_pred = knc.predict(X_test)


# Classification Evaluation
plot_cf(confusion_matrix(y_test, y_pred), "K-Nearest Neighbors")
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Store trained model
pickle.dump(knc, open('models/knc.pkl', 'wb'))
