from sklearn import metrics
from copy import deepcopy
import numpy as np
import tenseal as ts
import phe as paillier

class local_model:

    '''
    def __init__(self, model, public_key, private_key):
        self.model = model
        self.public_key = public_key
        self.private_key = private_key
    '''
    def __init__(self, model, public_key, private_key):
        self.model = model
        self.public_key = public_key
        self.private_key = private_key

    def evaluate(self, predicted, y_test):
        # evaluate model and print performance metrics
        print("\t<===> Model Performance Metrics: \n\t=============================")
        print("\t\t==> Accuracy: ", round(metrics.accuracy_score(y_test, predicted) * 100, 2), "%")
        print("\t\t==> Precision: ", round(metrics.precision_score(y_test, predicted) * 100, 2), "%")
        print("\t\t==> Recall: ", round(metrics.recall_score(y_test, predicted) * 100, 2), "%")
        print("\t\t==> F1 Score: ", round(metrics.f1_score(y_test, predicted) * 100, 2), "%")
        print("\t\t==> Specificity: ", round(metrics.recall_score
                                             (y_test, predicted, pos_label=0) * 100, 2), "%")
        print("\t\t==> Negative Predictive Value (NPV): ",
              round(metrics.precision_score(y_test, predicted, pos_label=0) * 100, 2), "%")

    def get_parameters(self):
        model_type = type(self.model).__name__
        if model_type == "SVC":
            parameters = self.svm_parameters()
        elif model_type == "LogisticRegression":
            parameters = self.lr_parameters()
        elif model_type == "GaussianNB":
            parameters = self.gnb_parameters()
        elif model_type == "SGDClassifier":
            parameters = self.sgd_parameters()
        elif model_type == "MLPClassifier":
            parameters = self.mlp_parameters()

        else:
            print("The provided model is not supported yet with this framework")
            return
        return parameters

    def set_parameters(self, parameters):
        model_type = type(self.model).__name__
        new_model = deepcopy(self.model)
        # print(parameters)
        if model_type == "SVC":
            decrypted_support_vector = [[paillier.decrypt(coef, self.private_key, self.public_key)
                                       for coef in row] for row in parameters["aggregated_support_vectors"]]
            decrypted_coefficients = [[paillier.decrypt(coef, self.private_key, self.public_key)
                                       for coef in row] for row in parameters["aggregated_coefficients"]]
            decrypted_intercept = paillier.decrypt(parameters["aggregated_intercept"],
                                                   self.private_key, self.public_key)

            support_vector =  [[num / (1000000 * 1000000) for num in sublist] for sublist in decrypted_support_vector]
            coefs = [[num / (1000000 * 1000000) for num in sublist] for sublist in decrypted_coefficients]
            intercept = [num / (1000000 * 1000000) for num in decrypted_intercept]

            new_model.support_vectors_ = np.vstack(support_vector)
            new_model.dual_coef_ = np.vstack([np.array(row) for row in coefs])
            new_model.intercept_ = intercept

        elif model_type == "LogisticRegression":
            decrypted_coefficients = [[self.private_key.decrypt(coef) for coef in row] for row in
                               parameters["aggregated_coefficients"]]
            decrypted_intercept = [self.private_key.decrypt(cp) for cp in parameters["aggregated_intercept"]]

            coefs = [[num / (1000000 * 1000000) for num in sublist] for sublist in decrypted_coefficients]
            intercept = [num / (1000000 * 1000000) for num in decrypted_intercept]

            #print("Decrypted Average coefs: ", coefs)
            #print("Decrypted Average intercept: ", intercept)

            new_model.coef_ = coefs
            new_model.intercept_ = intercept

        elif model_type == "GaussianNB":
            decrypted_class_priors = [self.private_key.decrypt(cp) for cp in parameters["aggregated_class_priors"]]
            decrypted_theta = [[self.private_key.decrypt(theta) for theta in row] for row in parameters["aggregated_theta"]]
            decrypted_sigma = [[self.private_key.decrypt(sigma) for sigma in row] for row in parameters["aggregated_sigma"]]

            class_priors = [num / (1000000*1000000) for num in decrypted_class_priors]
            theta = [[num /  (1000000*1000000) for num in sublist] for sublist in decrypted_theta]
            sigma = [[num /  (1000000*1000000) for num in sublist] for sublist in decrypted_sigma]

            #print("Decrypted Average class priors: ", class_priors)
            #print("Decrypted Average theta: ", theta)
            #print("Decrypted Average sigma: ", sigma)

            # new_model.set_params(**self.model.get_params())
            # Set the aggregated class priors, means, and variances in the new model
            new_model.class_prior_ =class_priors
            new_model.theta_ = theta
            new_model._sigma = sigma
            # Compute the var_ attribute based on the aggregated _sigma attribute
            new_model.var_ = np.copy(new_model._sigma)
            new_model.var_[new_model.var_ < np.finfo(np.float64).tiny] = np.finfo(np.float64).tiny
            # Use the classes_ attribute from one of the individual models
            # new_model.classes_ = parameters["classes"]

        elif model_type == "SGDClassifier":
            decrypted_coefficients = [[self.private_key.decrypt(coef) for coef in row] for row in
                               parameters["aggregated_coefficients"]]
            decrypted_intercept = [self.private_key.decrypt(cp) for cp in parameters["aggregated_intercept"]]

            coefs = [[num / (1000000 * 1000000) for num in sublist] for sublist in decrypted_coefficients]
            intercept = [num / (1000000 * 1000000) for num in decrypted_intercept]

            #print("Decrypted Average coefs: ", coefs)
            #print("Decrypted Average intercept: ", intercept)

            new_model.coef_ = coefs
            new_model.intercept_ = intercept

        elif model_type == "MLPClassifier":
            decrypted_coefficients = [[[self.private_key.decrypt(v) for v in row]
                                for row in coef] for coef in parameters["aggregated_coefficients"]]
            decrypted_intercepts = [[self.private_key.decrypt(v) for v in intercept]
                                    for intercept in parameters["aggregated_intercepts"]]

            coefs = [[[num / (1000000 * 1000000) for num in sublist] for sublist in row] for row in decrypted_coefficients]
            intercepts = [[num / (1000000 * 1000000) for num in sublist] for sublist in decrypted_intercepts]

            #print("\n\nDecrypted Average coefs: ", coefs)
            #print("\n\nDecrypted Average intercept: ", intercepts)

            new_model.coefs_ = coefs
            new_model.intercepts_ = intercepts

        return new_model

    def svm_parameters(self):
        support_vector = self.model.support_vectors_
        coefs = self.model.dual_coef_
        intercept = self.model.intercept_

        '''
        print("\n\nsupport vectors: ", support_vector)
        print("\n\ncoefs: ", coefs)
        print("\n\nintercept: ", intercept)
        '''
        support_vector_HE = [[0 for _ in range(len(support_vector[0]))] for _ in range(len(support_vector))]
        coefs_HE = [[0 for _ in range(len(coefs[0]))] for _ in range(len(coefs))]
        intercept_HE = [0 for _ in range(len(intercept))]

        '''
        for x in support_vector:
            for y in x:
                print("second dimension: ", y)
            print("\n finished a line \n ")
        '''

        pic = 0
        bas = 0
        #print("\n\nsize of support vectors")
        for x_index, x in enumerate(support_vector):
            pic+=1
            bas = 0
            for y_index, y in enumerate(x):
                bas+=1
            #print("size of second dimension: ", bas)
        #print("size of 1st dimension: ", pic)

        for x_index, x in enumerate(support_vector):
            for y_index, y in enumerate(x):
                support_vector_HE[x_index][y_index] = self.public_key.encrypt(y)

        for x_index, x in enumerate(coefs):
                    for y_index, y in enumerate(x):
                        coefs_HE[x_index][y_index] = self.public_key.encrypt(y)

        for intercept_index, intercept in enumerate(intercept):
            intercept_HE[intercept_index] = self.public_key.encrypt(intercept)

        pic = 0
        bas = 0
        #print("\n\nsize of support vectors")
        for x_index, x in enumerate(support_vector_HE):
            pic+=1
            bas = 0
            for y_index, y in enumerate(x):
                bas+=1
            #print("size of second dimension: ", bas)
        #print("size of 1st dimension: ", pic)

        
        return {"support_vector": support_vector_HE, "coefs": coefs_HE, "intercept": intercept_HE}

    def lr_parameters(self):
        coefs = self.model.coef_
        intercept = self.model.intercept_
        coefs_HE = [[0 for _ in range(len(coefs[0]))] for _ in range(len(coefs))]
        intercept_HE = [0 for _ in range(len(intercept))]

        for x_index, x in enumerate(coefs):
            for y_index, y in enumerate(x):
                coefs_HE[x_index][y_index] = self.public_key.encrypt(y * 1000000)

        for x_index, x in enumerate(intercept):
            intercept_HE[x_index] = self.public_key.encrypt(x * 1000000)

        return {"coefs": coefs_HE, "intercept": intercept_HE}

    def gnb_parameters(self):
        class_priors = self.model.class_prior_
        theta = self.model.theta_
        sigma = self.model.sigma_

        #print("class priros: ", class_priors)
        #print("theta: ", theta)
        #print("sigma: ", sigma)

        class_priors_HE = [0 for _ in range(len(class_priors))]
        theta_HE = [[0 for _ in range(len(theta[0]))] for _ in range(len(theta))]
        sigma_HE = [[0 for _ in range(len(sigma[0]))] for _ in range(len(sigma))]

        for x_index, x in enumerate(class_priors):
            class_priors_HE[x_index] = self.public_key.encrypt(x*1000000)

        for x_index, x in enumerate(theta):
            for y_index, y in enumerate(x):
                theta_HE[x_index][y_index] =  self.public_key.encrypt(y*1000000)

        for x_index, x in enumerate(sigma):
            for y_index, y in enumerate(x):
                sigma_HE[x_index][y_index] =  self.public_key.encrypt(y*1000000)

        return {"class_priors": class_priors_HE, "theta": theta_HE, "sigma": sigma_HE}

    def sgd_parameters(self):
        coefs = self.model.coef_
        intercept = self.model.intercept_

        #print("\n\ncoefs: ", coefs)
        #print("\n\nintercept: ", intercept)

        coefs_HE = [[0 for _ in range(len(coefs[0]))] for _ in range(len(coefs))]
        intercept_HE = [0 for _ in range(len(intercept))]

        for x_index, x in enumerate(coefs):
            for y_index, y in enumerate(x):
                coefs_HE[x_index][y_index] = self.public_key.encrypt(y * 1000000)

        for x_index, x in enumerate(intercept):
            intercept_HE[x_index] = self.public_key.encrypt(x * 1000000)

        return {"coefs": coefs_HE, "intercept": intercept_HE}

    def mlp_parameters(self):
        coefs = self.model.coefs_
        intercept = self.model.intercepts_

        #print("\ncoefs: ", coefs)
        #print("\nintercept: ", intercept)

        coefs_HE = []
        for layer in coefs:
            layer_coefs_HE = []
            for neuron_coefs in layer:
                neuron_coefs_HE = [self.public_key.encrypt(coef * 1000000) for coef in neuron_coefs]
                layer_coefs_HE.append(neuron_coefs_HE)
            coefs_HE.append(layer_coefs_HE)

        intercepts_HE = [[self.public_key.encrypt(intercept * 1000000) for intercept in layer] for layer in
                         intercept]

        return {"coefs": coefs_HE, "intercept": intercepts_HE}
