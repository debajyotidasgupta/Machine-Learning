from sklearn.decomposition import PCA
import data_processor
import model
import utils

if __name__ == '__main__':
    filename = 'Data/Train_B.csv'
    dataset = load_csv(filename)
    cols = dataset.pop(0)

    dataset = dataset[:len(dataset)//100]
    post_process_dataset = prepare(dataset,cols)
    X_train, X_test = train_test_split(dataset)

    n_folds = 5
    scores, summary = evaluate_algorithm(X_train, naive_bayes, n_folds)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))