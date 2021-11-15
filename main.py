from src.data_preparation import data_preparation

if __name__ == '__main__':

    train_data, eval_data = data_preparation('dataset/application_train.csv', 'dataset/application_test.csv')

