from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    penguin_data = pd.read_csv('./penguins.csv').dropna()

    normalized_sex = preprocessing.OrdinalEncoder().fit_transform(penguin_data[penguin_data["species"]=="Adelie"].sex.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        penguin_data[penguin_data["species"]=="Adelie"]["body_mass_g"],
        normalized_sex,
        train_size=0.7,
        random_state=87234
    )

    clf = linear_model.LogisticRegression(random_state=0).fit(X_train.values.reshape(-1, 1), y_train)

    line_xs = np.arange(3000, 5000, 100)#line_xs = np.arange(2500, 5000, 100)

    #plt.scatter(X_train,y_train)
    plt.scatter(X_test, y_test)

    plt.plot(line_xs, clf.predict_proba(line_xs.reshape((-1, 1)))[:,1])

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
