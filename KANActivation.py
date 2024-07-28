from pykan import BSplineActivation, PolynomialActivation, RBFActivation, WaveletActivation

def get_activation_function(name):
    if name == 'b_spline':
        return BSplineActivation()
    if name == 'polynomial':
        return PolynomialActivation()
    if name == 'rbf':
        return RBFActivation()
    if name == 'wavelet':
        return WaveletActivation()
    else:
        raise ValueError("Unsupported activation function")

activation_functions = ['b_spline', 'polynomial', 'rbf', 'wavelet']
