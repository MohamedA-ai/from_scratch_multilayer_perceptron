from sklearn.datasets import load_iris
import pandas as pd

if __name__ == '__main__':
    iris_data = load_iris()
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['target'] = iris_data.target
    df.to_csv('iris.csv', index=False)
