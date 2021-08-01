"""
anchor.py



Inspired by https://github.com/marcotcr/anchor
"""

# How Anchor works:
#
# black box model f: X -> Y
# instance x elemnt of X
# f(x) = y individual prediction 
# 
# perturbation distribution D
# 
# Let A be a rule (set of predicates), such that A(x) return 1 if
# all its feature predicates are true for instance x
#
# A is an Anchor if 


class AnchorUTS:        

    """
    Performs Anchor on univariate time series
    """

    def __init__(self):
        pass


    def explain(self, timeseries_data, y_true, model):

        _timeseries_data = np.array(timeseries_data)
        _y_true = np.array(y_true)

        if _timeseries_data.shape != _y_true.shape:
            raise Exception('Time series data and labels are not of the same shape')

        explanations = [
            self.explain_instance(_timeseries_data[i], y_true[i], model)
            for i in range(_timeseries_data.shape[0])
        ]

    def explain_instance(self, timeseries_instance, true_class, model):





        pass