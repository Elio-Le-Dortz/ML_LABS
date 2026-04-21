import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.
    means = np.mean(train_features, axis = 0)
    std = np.std(train_features, axis = 0)
    std[std==0] = 1
    train_features = normalize_fn(train_features, means, std)  ##Normalization of the training data.
    test_features = normalize_fn(test_features,means,std)     ##Normalization of the test data.

    ### WRITE YOUR CODE HERE to do any other data processing
    if args.method != "knn":
        train_features = append_bias_term(train_features)
        test_features = append_bias_term(test_features)

    ## 3. K-Fold cross-validation (when not evaluating on test set)

    if not args.test:
        n_samples = train_features.shape[0]
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // args.n_folds

        cv_accs, cv_f1s, cv_mses = [], [], []

        for fold in range(args.n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < args.n_folds - 1 else n_samples

            val_idx   = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            fold_train_X          = train_features[train_idx]
            fold_val_X            = train_features[val_idx]
            fold_train_y_reg      = train_labels_reg[train_idx]
            fold_val_y_reg        = train_labels_reg[val_idx]
            fold_train_y_classif  = train_labels_classif[train_idx]
            fold_val_y_classif    = train_labels_classif[val_idx]

            if args.method == "dummy_classifier":
                method_obj = DummyClassifier(arg1=1, arg2=2)
            elif args.method == "knn":
                method_obj = KNN(args.K, args.task)
            elif args.method == "logistic_regression":
                method_obj = LogisticRegression(args.lr, args.max_iters)
            elif args.method == "linear_regression":
                method_obj = LinearRegression(args.regularization_param)
            else:
                raise ValueError(f"Unknown method: {args.method}")

            if args.task == "classification":
                assert args.method != "linear_regression", f"You should use linear regression as a regression method"
                method_obj.fit(fold_train_X, fold_train_y_classif)
                preds = method_obj.predict(fold_val_X)
                cv_accs.append(accuracy_fn(preds, fold_val_y_classif))
                cv_f1s.append(macrof1_fn(preds, fold_val_y_classif))
            elif args.task == "regression":
                assert args.method != "logistic_regression", f"You should use logistic regression as a classification method"
                method_obj.fit(fold_train_X, fold_train_y_reg)
                preds = method_obj.predict(fold_val_X)
                cv_mses.append(mse_fn(preds, fold_val_y_reg))
            else:
                raise ValueError(f"Unknown task: {args.task}")

        if args.task == "classification":
            print(f"\n{args.n_folds}-Fold CV: accuracy = {np.mean(cv_accs):.3f}% - F1-score = {np.mean(cv_f1s):.6f}")
        elif args.task == "regression":
            print(f"\n{args.n_folds}-Fold CV: MSE = {np.mean(cv_mses):.6f}")

        return

    ## 4. Initialize the method and train/evaluate on the test set

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj  = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(args.K,args.task)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(args.lr,args.max_iters)

    elif args.method == "linear_regression":
        method_obj = LinearRegression(args.regularization_param)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    if args.task == "classification":
        assert args.method != "linear_regression", f"You should use linear regression as a regression method"
        # Fit the method on training data
        preds_train = method_obj.fit(train_features, train_labels_classif)

        # Predict on unseen data
        preds = method_obj.predict(test_features)

        # Report results: performance on train and test sets
        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    elif args.task == "regression":
        assert args.method != "logistic_regression", f"You should use logistic regression as a classification method"
        # Fit the method on training data
        preds_train = method_obj.fit(train_features, train_labels_reg)

        # Predict on unseen data
        preds = method_obj.predict(test_features)

        # Report results: MSE on train and test sets
        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / linear_regression",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of neighboring datapoints used for knn",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    parser.add_argument(
        "--regularization_param",
        type=float,
        default=0,
        help="regularization parameter for linear regression",
    )

    parser.add_argument("--n_folds",
                        type=int,
                        default=5,
                        help="number of folds for cross-validation (used when --test is not set)",
                       )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)
