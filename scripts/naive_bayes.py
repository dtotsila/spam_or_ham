from utils import read_data, plot_cf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import pickle

# Read Train Test data
X_train, y_train = read_data('train')
X_test, y_test = read_data('test')

# init classifier and train
nb = MultinomialNB().fit(X_train, y_train)

# test
y_pred = nb.predict(X_test)

# Classification Evaluation
plot_cf(confusion_matrix(y_test, y_pred), "Naive Bayes")
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Store trained model
pickle.dump(nb, open('models/nb.pkl', 'wb'))
