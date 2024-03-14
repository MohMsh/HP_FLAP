from copy import deepcopy
import numpy as np
from sklearn import metrics


class global_model:

    def __init__(self, models_parameters, global_model_, x_test, y_test, current_round, print_model_summary,
                 print_model_performance, public_key):
        self.models_parameters = models_parameters
        self.x_test = x_test
        self.y_test = y_test
        self.global_model_ = global_model_
        self.current_round = current_round
        self.print_model_summary = print_model_summary
        self.print_model_performance = print_model_performance
        self.public_key = public_key

    def print_round(self):
        print("\n\t==========================================\n"
              "\tGlobal model aggregation initiated (Round ", self.current_round, ")",
              "\n\t==========================================\n")

    def aggregate(self):
        # print some info related to global model
        self.print_round()
        model_type = type(self.global_model_).__name__
        if model_type == "SVC":
            # self.global_model_, aggregated_parameters = self.svm_aggregate()
            aggregated_parameters = self.svm_aggregate()
            # predicted = self.svm_custom_predict(self.global_model_, self.x_test)
            # self.evaluate(predicted) if self.print_model_performance else None
            # self.svm_summary(self.global_model_) if self.print_model_summary else None
        elif model_type == "LogisticRegression":
            # self.global_model_, aggregated_parameters = self.lr_aggregate()
            aggregated_parameters = self.lr_aggregate()
            # predicted = self.global_model_.predict(self.x_test)
            # self.evaluate(predicted) if self.print_model_performance else None
            # self.lr_summary(self.global_model_) if self.print_model_summary else None
        elif model_type == "GaussianNB":
            # self.global_model_, aggregated_parameters = self.gnb_aggregate()
            aggregated_parameters = self.gnb_aggregate()
            # predicted = self.global_model_.predict(self.x_test)
            # self.evaluate(predicted) if self.print_model_performance else None
            # self.gnb_summary(self.global_model_) if self.print_model_summary else None
        elif model_type == "SGDClassifier":
            # self.global_model_, aggregated_parameters = self.sgd_aggregate()
            aggregated_parameters = self.sgd_aggregate()
            # predicted = self.global_model_.predict(self.x_test)
            # self.evaluate(predicted) if self.print_model_performance else None
            # self.sgd_summary(self.global_model_) if self.print_model_summary else None
        elif model_type == "MLPClassifier":
            # self.global_model_, aggregated_parameters = self.mlp_aggregate()
            aggregated_parameters = self.mlp_aggregate()
            # predicted = self.global_model_.predict(self.x_test)
            # self.evaluate(predicted) if self.print_model_performance else None
            # self.mlp_summary(self.global_model_) if self.print_model_summary else None
        else:
            print("The provided model is not supported yet with this framework")
            return

        # return self.global_model_, aggregated_parameters
        return aggregated_parameters

    def evaluate(self, predicted):
        print("\t<===> Model Performance Metrics: \n\t=============================")
        # evaluate model and print performance metrics
        print("\t\t==> Accuracy: ", round(metrics.accuracy_score(self.y_test, predicted) * 100, 2), "%")
        print("\t\t==> Precision: ", round(metrics.precision_score(self.y_test, predicted) * 100, 2), "%")
        print("\t\t==> Recall: ", round(metrics.recall_score(self.y_test, predicted) * 100, 2), "%")
        print("\t\t==> F1 Score: ", round(metrics.f1_score(self.y_test, predicted) * 100, 2), "%")
        print("\t\t==> Specificity: ", round(metrics.recall_score
                                             (self.y_test, predicted, pos_label=0) * 100, 2), "%")
        print("\t\t==> Negative Predictive Value (NPV): ",
              round(metrics.precision_score(self.y_test, predicted, pos_label=0) * 100, 2), "%")
        print("\t=============================")

    def svm_aggregate(self):
        # Initialize the aggregated support vectors, coefficients, and intercept
        aggregated_support_vectors = [row[:] for row in self.models_parameters[0]["support_vector"]]
        aggregated_coefficients = [row[:] for row in self.models_parameters[0]["coefs"]]
        aggregated_intercept = self.models_parameters[0]["intercept"][0]

        # Combine support vectors, coefficients, and intercepts from each client's model
        for parameter in self.models_parameters[1:]:
            support_vector = parameter["support_vector"]
            coefs = parameter["coefs"]
            intercept = parameter["intercept"]

            for i, sv in enumerate(support_vector):
                for j, x in enumerate(sv):
                    aggregated_support_vectors[i][j] = aggregated_support_vectors[i][j] + x

            for i, coef in enumerate(coefs):
                for j, x in enumerate(coef):
                    aggregated_coefficients[i][j] = aggregated_coefficients[i][j] + x

            aggregated_intercept[i] = aggregated_intercept[i] + intercept[0]

        # Average calculation
        reciprocal = int((1.0 / len(self.models_parameters) * 1000000))

        for i, sv in enumerate(aggregated_support_vectors):
            for j, x in enumerate(sv):
                aggregated_support_vectors[i][j] = aggregated_support_vectors[i][j] * reciprocal

        for i, coef in enumerate(aggregated_coefficients):
            for j, x in enumerate(coef):
                aggregated_coefficients[i][j] = aggregated_coefficients[i][j] * reciprocal

        for i, intercept in enumerate(aggregated_intercept):
            aggregated_intercept[i] = aggregated_intercept[i] * reciprocal

        return {"aggregated_support_vectors":aggregated_support_vectors ,
                "aggregated_coefficients": aggregated_coefficients,
                "aggregated_intercept": aggregated_intercept}

    # this function is for HE aggregation
    def lr_aggregate(self):
        # Initialize the aggregated class priors, means, and variances
        aggregated_coefficients =  [row[:] for row in self.models_parameters[0]["coefs"]]
        aggregated_intercept = self.models_parameters[0]["intercept"]

        for parameter in self.models_parameters[1:]:
            coefs = parameter["coefs"]
            intercepts = parameter["intercept"]

            for i, coef in enumerate(coefs):
                for j, x in enumerate(coef):
                    aggregated_coefficients[i][j] = aggregated_coefficients[i][j] + x

            for i, intercept in enumerate(intercepts):
                aggregated_intercept[i] = aggregated_intercept[i] + intercept

        reciprocal = int((1.0 / len(self.models_parameters) * 1000000))

        for i, coef in enumerate(aggregated_coefficients):
            for j, x in enumerate(coef):
                aggregated_coefficients[i][j] = aggregated_coefficients[i][j] * reciprocal

        for i, intercept in enumerate(aggregated_intercept):
            aggregated_intercept[i] = aggregated_intercept[i] * reciprocal

        return {"aggregated_coefficients": aggregated_coefficients, "aggregated_intercept": aggregated_intercept}

    def gnb_aggregate(self):
        # Initialize the aggregated class priors, means, and variances
        aggregated_class_priors = self.models_parameters[0]["class_priors"]
        aggregated_theta = [row[:] for row in self.models_parameters[0]["theta"]]
        aggregated_sigma = [row[:] for row in self.models_parameters[0]["sigma"]]

        for parameter in self.models_parameters[1:]:
            priors = parameter["class_priors"]
            thetas = parameter["theta"]
            sigmas = parameter["sigma"]

            for i, class_prior in enumerate(priors):
                aggregated_class_priors[i] = aggregated_class_priors[i] + class_prior

            for i, theta in enumerate(thetas):
                for j, x in enumerate(theta):
                    aggregated_theta[i][j] = aggregated_theta[i][j] + x

            for i, sigma in enumerate(sigmas):
                for j, x in enumerate(sigma):
                    aggregated_sigma[i][j] = aggregated_sigma[i][j] + x

        reciprocal = int((1.0 / len(self.models_parameters) * 1000000))

        for i, class_prior in enumerate(aggregated_class_priors):
            aggregated_class_priors[i] = aggregated_class_priors[i] * reciprocal
        for i, thetas in enumerate(aggregated_theta):
            for j, x in enumerate(thetas):
                aggregated_theta[i][j] = aggregated_theta[i][j] * reciprocal
        for i, sigmas in enumerate(aggregated_sigma):
            for j, x in enumerate(sigmas):
                aggregated_sigma[i][j] = aggregated_sigma[i][j] * reciprocal

        return {"aggregated_class_priors": aggregated_class_priors, "aggregated_theta": aggregated_theta,
                "aggregated_sigma": aggregated_sigma}

    def sgd_aggregate(self):
        # Initialize the aggregated class priors, means, and variances
        aggregated_coefficients = [row[:] for row in self.models_parameters[0]["coefs"]]
        aggregated_intercept = self.models_parameters[0]["intercept"]

        for parameter in self.models_parameters[1:]:
            coefs = parameter["coefs"]
            intercepts = parameter["intercept"]

            for i, coef in enumerate(coefs):
                for j, x in enumerate(coef):
                    aggregated_coefficients[i][j] = aggregated_coefficients[i][j] + x

            for i, intercept in enumerate(intercepts):
                aggregated_intercept[i] = aggregated_intercept[i] + intercept

        reciprocal = int((1.0 / len(self.models_parameters) * 1000000))

        for i, coef in enumerate(aggregated_coefficients):
            for j, x in enumerate(coef):
                aggregated_coefficients[i][j] = aggregated_coefficients[i][j] * reciprocal

        for i, intercept in enumerate(aggregated_intercept):
            aggregated_intercept[i] = aggregated_intercept[i] * reciprocal

        return {"aggregated_coefficients": aggregated_coefficients, "aggregated_intercept": aggregated_intercept}

    def mlp_aggregate(self):
        aggregated_coefficients = [[[coef for coef in neuron_coefs] for neuron_coefs in layer_coefs] for layer_coefs in
                                   self.models_parameters[0]["coefs"]]
        aggregated_intercepts = [[intercept for intercept in layer_intercepts] for layer_intercepts in
                                self.models_parameters[0]["intercept"]]

        # Combine coefficients and intercepts from each client's model starting from the second model
        for parameter in self.models_parameters[1:]:
            coefs = parameter["coefs"]
            intercepts = parameter["intercept"]


            for i, coef in enumerate(coefs):
                for j, y in enumerate(coef):
                    for k, v in enumerate(y):
                        aggregated_coefficients[i][j][k] = aggregated_coefficients[i][j][k] + v

            for i, intercept in enumerate(intercepts):
                for j, x in enumerate(intercept):
                    aggregated_intercepts[i][j] = aggregated_intercepts[i][j] + x

        # "Average" the coefficients and intercept by multiplying by the reciprocal of the number of models
        reciprocal = int((1.0 / len(self.models_parameters) * 1000000))

        for i, coef in enumerate(coefs):
            for j, y in enumerate(coef):
                for k, v in enumerate(y):
                    aggregated_coefficients[i][j][k] = aggregated_coefficients[i][j][k] * reciprocal

        for i, intercept in enumerate(intercepts):
            for j, x in enumerate(intercept):
                aggregated_intercepts[i][j] = aggregated_intercepts[i][j] * reciprocal

        return {"aggregated_coefficients": aggregated_coefficients, "aggregated_intercepts": aggregated_intercepts}

    def svm_rbf_kernel(self, X1, X2, gamma):
        sq_dist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sq_dist)

    def svm_linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def svm_poly_kernel(self, X1, X2, degree, gamma, coef0):
        return (gamma * np.dot(X1, X2.T) + coef0) ** degree

    def svm_sigmoid_kernel(self, X1, X2, gamma, coef0):
        return np.tanh(gamma * np.dot(X1, X2.T) + coef0)

    def svm_custom_decision_function(self, model, X):
        if model.kernel == 'linear':
            kernel_matrix = self.svm_linear_kernel(X, model.support_vectors_)
        elif model.kernel == 'poly':
            kernel_matrix = self.svm_poly_kernel(X, model.support_vectors_, model.degree, model.gamma, model.coef0)
        elif model.kernel == 'rbf':
            kernel_matrix = self.svm_rbf_kernel(X, model.support_vectors_, model.gamma)
        elif model.kernel == 'sigmoid':
            kernel_matrix = self.svm_sigmoid_kernel(X, model.support_vectors_, model.gamma, model.coef0)
        elif model.kernel == 'precomputed':
            raise NotImplementedError("Custom decision function doesn't support 'precomputed' kernel")
        else:
            raise ValueError("Unknown kernel type")

        decision_values = np.dot(kernel_matrix, model.dual_coef_.T) + model.intercept_
        return decision_values

    def svm_custom_predict(self, model, X):
        decision_values = self.svm_custom_decision_function(model, X)
        if len(model.classes_) == 2:
            y_pred = np.where(decision_values >= 0, model.classes_[1], model.classes_[0])
        else:
            y_pred = model.classes_[np.argmax(decision_values, axis=1)]

        return y_pred.ravel()

    def svm_summary(self, model):
        print("\tSupport Vector Machine Model Summary\n\t=============================")
        print("\t\t==> Model type:", type(model).__name__)  # Output: "SVC"
        print(f"\t\t==> Kernel: {model.kernel}")

        if model.kernel == 'poly':
            print(f"\t\t==> Degree: {model.degree}")

        if model.kernel in ['poly', 'rbf', 'sigmoid']:
            print(f"\t\t==> Gamma: {'auto' if model.gamma == 'auto' else model._gamma}")

        if model.kernel in ['poly', 'sigmoid']:
            print(f"\t\t==> Coef0: {model.coef0}")

        print(f"\t\t==> C (Regularization parameter): {model.C}")
        print(f"\t\t==> Shrinking: {model.shrinking}")
        print(f"\t\t==> Probability estimates: {model.probability}")
        print(f"\t\t==> Tolerance: {model.tol}")
        try:
            print(f"\t\t==> Class labels: {model.classes_}")
            print(f"\t\t==> Number of support vectors: {model.n_support_}")
            print(f"\t\t==> Intercept: {model.intercept_} \n\t=============================\n")
        except BaseException as e:
            pass

    def lr_summary(self, model):
        print("\tLogistic Regression Model Summary\n\t=============================")
        print("\t\t==> Model type:", type(model).__name__)  # Output: "LogisticRegression"
        print(f"\t\t==> Solver: {model.solver}")
        print(f"\t\t==> Penalty: {model.penalty}")
        print(f"\t\t==> C (Inverse regularization strength): {model.C}")
        print(f"\t\t==> Fit intercept: {model.fit_intercept}")
        print(f"\t\t==> Max iterations: {model.max_iter}")
        print(f"\t\t==> Tolerance: {model.tol}")

        try:
            print(f"\t\t==> Class labels: {model.classes_}")
            print(f"\t\t==> Coefficients: {model.coef_}")
            print(f"\t\t==> Intercept: {model.intercept_} \n\t=============================\n")
        except BaseException as e:
            pass

    def gnb_summary(self, model):
        print("\tGaussian Naive Bayes Model Summary\n\t=============================")
        print("\t\t==> Model type:", type(model).__name__)  # Output: "GaussianNB"
        print(f"\t\t==> Variance smoothing: {model.var_smoothing}")

        try:
            print(f"\t\t==> Class labels: {model.classes_}")
            print(f"\t\t==> Class priors: {model.class_prior_}")
            print(f"\t\t==> Class counts: {model.class_count_}")
            print(f"\t\t==> Mean: {model.theta_}")
            print(f"\t\t==> Variance: {model.sigma_} \n\t=============================\n")
        except BaseException as e:
            pass

    def sgd_summary(self, model):
        print("\tSGD Classifier Model Summary\n\t=============================")
        print("\t\t==> Model type:", type(model).__name__)  # Output: "SGDClassifier"
        print(f"\t\t==> Loss: {model.loss}")
        print(f"\t\t==> Penalty: {model.penalty}")
        print(f"\t\t==> Alpha: {model.alpha}")
        print(f"\t\t==> L1 ratio: {model.l1_ratio}")
        print(f"\t\t==> Fit intercept: {model.fit_intercept}")
        print(f"\t\t==> Max iterations: {model.max_iter}")
        print(f"\t\t==> Tolerance: {model.tol}")
        print(f"\t\t==> Learning rate: {model.learning_rate}")
        print(f"\t\t==> Eta0: {model.eta0}")

        try:
            print(f"\t\t==> Class labels: {model.classes_}")
            print(f"\t\t==> Coefficients: {model.coef_}")
            print(f"\t\t==> Intercept: {model.intercept_} \n\t=============================\n")
        except BaseException as e:
            pass

    def mlp_summary(self, model):
        print("\tMLP Model Summary\n\t=============================")
        print("\t\t==> Model type:", type(model).__name__)  # Output: "MLPClassifier" or "MLPRegressor"

        print("\t\t==> Activation function:", model.activation)
        print("\t\t==> Solver (optimizer):", model.solver)
        print("\t\t==> Alpha (L2 regularization):", model.alpha)
        print("\t\t==> Learning rate:", model.learning_rate)
        print("\t\t==> Initial learning rate (eta0):", model.learning_rate_init)

        # Print hidden layer sizes
        print("\t\t==> Hidden layer sizes:", model.hidden_layer_sizes)

        # Print weights and biases for each layer
        print("\t\t==> Layer weights and biases:")
        for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
            print(f"\t\t==> Layer {i + 1}:")
            print(f"\t\t\tWeights: {coef}")
            print(f"\t\t\tBiases: {intercept}")
        print("\n\t=============================\n")
