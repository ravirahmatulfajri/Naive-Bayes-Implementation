from collections import defaultdict
from math import log, exp

class NaiveBayesClassifier():
    """Multionomial Naive Bayes classifier with Laplace Smoothing.
    
    Methods:
        train() - fits the model to a training dataset
        predict() - predicts class labels for test data
    """

    def __init__(self):
        self.priors = None
        self.posteriors = None


    def train(self, X, y):
        """Fits training instances X and class labels y to a supervised Naive Bayes model.
        Uses Laplace Smoothing to make all posterior probabilities non-zero.
        """

        def _calculate_priors(class_list):
            """calculate_priors takes in a list of class labels in String format.
            It calculates the class priors and fits them to the classifier.
            """ 
            
            prior_dd = defaultdict(int)
            # Count all class labels and add to dictionary
            for class_label in class_list:
                prior_dd[class_label] += 1
            self.priors = dict(prior_dd)


        def _calculate_posteriors(priors_dict, instance_list, class_list):
            """calculate_posteriors takes in a dictionary or prior probabilities for class labels,
            a 2D list of instance values for each attribute and a list of class labels.
            
            It calculates and fits posterior probabilities to the classifier.
            """

            # Initialize a list of dictionaries, one for each attribute
            posteriors_dictlist = [dict() for x in range(len(instance_list))]

            # Initialize a default dict for each class label, for each attribute dictionary
            for attribute_dd in posteriors_dictlist:
                for class_label in priors_dict.keys():
                    # Start at 1 for Laplace smoothing
                    attribute_dd[class_label] = defaultdict(lambda:1)


            # Count the number of instances for each conditional probability P(Attribute=attr_instance | Class)
            for col in range(len(instance_list)):
                for row in range(len(instance_list[col])):
                    
                    #Format is <Attribute> <Class> <Attribute Instance>
                    posteriors_dictlist[col][class_list[row]][instance_list[col][row]] += 1
                
                # Keep track of all attribute possibilites
                attr_set = set()
                for label in posteriors_dictlist[col].keys():
                    for attr in posteriors_dictlist[col][label].keys():
                        attr_set.add(attr)
                
                # Add attributes with counts of 1 (Laplace Smoothing) when no occurances for a given class
                for label in posteriors_dictlist[col].keys():
                    for attr in attr_set:
                        if attr not in posteriors_dictlist[col][label].keys():
                            # Start at 1 for Laplace smoothing
                            posteriors_dictlist[col][label][attr] = 1
                            
            self.posteriors = posteriors_dictlist


        # Fit prior and posterior probabilities to the model
        _calculate_priors(y)
        _calculate_posteriors(self.priors, X, y)


    def predict(self, test_set):
        """Predicts the class for a set of instances using a trained supervised Naive Bayes model.
        Returns a list of class predictions.
        """
        
        if (self.priors is None or self.posteriors is None):
            raise ValueError("Naive Models model has not been fit.")

        predictions = []
        n_test_instances = len(test_set[0])

        # Make a prediction for every instance in the test set
        for test_row in range(n_test_instances):
            label_predict_probs = []
            
            # Calculate prediction probability for each class label
            for label in self.priors.keys():
                label_count = self.priors[label]
                
                # Prior log probability log(P(label))
                label_prob = log(label_count / n_test_instances)
                
                # Sum the prediction probability and log(posterior probabilities) to avoid underlow
                # Dividing by the number of labels + number of attribute values (Laplace Smoothing)
                for test_col in range(len(test_set)):
                    attr = test_set[test_col][test_row]
                    
                    posterior_prob = self.posteriors[test_col][label][attr] / \
                            (label_count + len(self.posteriors[test_col][label]))
                    
                    label_prob += log(posterior_prob)
                
                # Turn log probabilitiy back in probability
                label_prob = exp(label_prob)
                label_predict_probs.append((label_prob, label))

            # Sort the predictions from high-low and predict the label with the highest probability
            label_predict_probs.sort(reverse=True)
            predictions.append(label_predict_probs[0][1])
        
        return predictions