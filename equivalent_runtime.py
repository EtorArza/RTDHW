correction_coefficients = [1.0,0.999,0.998,0.997,0.996,0.995,0.994,0.993,0.992,0.991,0.99,0.989,0.988,0.987,0.986,0.985,0.984,0.983,0.982,0.981,0.98,0.979,0.978,0.977,0.976,0.975,0.974,0.973,0.972,0.971,0.97,0.969,0.968,0.967,0.966,0.965,0.964,0.963,0.962,0.961,0.96,0.959,0.958,0.957,0.956,0.955,0.954,0.953,0.952,0.951,0.95,0.948,0.946,0.944,0.942,0.94,0.938,0.9359999999999999,0.9339999999999999,0.9319999999999999,0.9299999999999999,0.9279999999999999,0.9259999999999999,0.9239999999999999,0.9219999999999999,0.9199999999999999,0.9179999999999999,0.9159999999999999,0.9139999999999999,0.9119999999999999,0.9099999999999999,0.9079999999999999,0.9059999999999999,0.9039999999999999,0.9019999999999999,0.8999999999999999,0.8979999999999999,0.8959999999999999,0.894,0.892,0.89,0.888,0.886,0.884,0.882,0.88,0.878,0.876,0.874,0.872,0.87,0.868,0.866,0.864,0.862,0.86,0.858,0.856,0.854,0.852,0.85,0.8428571428571429,0.8357142857142857,0.8285714285714285,0.8214285714285714,0.8142857142857143,0.8071428571428572,0.7999999999999999,0.7928571428571428,0.7857142857142857,0.7785714285714286,0.7714285714285714,0.7642857142857142,0.7571428571428571,0.75,0.7428571428571429,0.7357142857142857,0.7285714285714285,0.7214285714285714,0.7142857142857143,0.7071428571428571,0.7,0.6928571428571428,0.6857142857142857,0.6785714285714286,0.6714285714285714,0.6642857142857143,0.6571428571428571,0.65,0.6428571428571428,0.6357142857142857,0.6285714285714286,0.6214285714285714,0.6142857142857143,0.6071428571428571,0.6,0.5928571428571429,0.5857142857142856,0.5785714285714285,0.5714285714285714,0.5642857142857143,0.5571428571428572,0.55,0.5428571428571429,0.5357142857142857,0.5285714285714286,0.5214285714285714,0.5142857142857142,0.5071428571428571,0.5]
percentages_higher_than_predicted = [0.5,0.4988839285714286,0.4966517857142857,0.4955357142857143,0.4938616071428571,0.4919084821428571,0.4910714285714286,0.4880022321428571,0.4868861607142857,0.4854910714285714,0.4846540178571429,0.4835379464285714,0.482421875,0.4799107142857143,0.4773995535714286,0.4768415178571429,0.4757254464285714,0.474609375,0.4723772321428571,0.4712611607142857,0.4690290178571429,0.4673549107142857,0.4662388392857143,0.4642857142857143,0.4620535714285714,0.4603794642857143,0.4547991071428571,0.4475446428571429,0.4428013392857143,0.4416852678571429,0.4408482142857143,0.4402901785714286,0.4391741071428571,0.4352678571428571,0.4324776785714286,0.4299665178571429,0.4288504464285714,0.4255022321428571,0.421875,0.4193638392857143,0.4176897321428571,0.4162946428571429,0.4151785714285714,0.4140625,0.4132254464285714,0.4095982142857143,0.4079241071428571,0.4079241071428571,0.4068080357142857,0.4051339285714286,0.404296875,0.3995535714285714,0.3970424107142857,0.3936941964285714,0.3892299107142857,0.3864397321428571,0.384765625,0.3833705357142857,0.3811383928571429,0.3777901785714286,0.3747209821428571,0.3699776785714286,0.3663504464285714,0.3627232142857143,0.3582589285714286,0.3529575892857143,0.3484933035714286,0.3454241071428571,0.3412388392857143,0.3387276785714286,0.3362165178571429,0.3306361607142857,0.32421875,0.3197544642857143,0.3166852678571429,0.3130580357142857,0.3102678571428571,0.3063616071428571,0.3030133928571429,0.3002232142857143,0.2974330357142857,0.2952008928571429,0.2932477678571429,0.2918526785714286,0.2887834821428571,0.2859933035714286,0.2826450892857143,0.2784598214285714,0.2759486607142857,0.2734375,0.2717633928571429,0.2700892857142857,0.2664620535714286,0.2633928571428571,0.259765625,0.2583705357142857,0.2561383928571429,0.251953125,0.2505580357142857,0.248046875,0.2449776785714286,0.2405133928571429,0.2352120535714286,0.2218191964285714,0.2081473214285714,0.197265625,0.1919642857142857,0.1788504464285714,0.1721540178571429,0.1651785714285714,0.1576450892857143,0.1481584821428571,0.1411830357142857,0.1336495535714286,0.1247209821428571,0.1177455357142857,0.1146763392857143,0.1107700892857143,0.103515625,0.0973772321428571,0.09375,0.0870535714285714,0.078125,0.0652901785714286,0.060825892857142905,0.0516183035714286,0.0440848214285714,0.033203125,0.02734375,0.024274553571428603,0.023158482142857095,0.022321428571428603,0.019252232142857095,0.015904017857142905,0.014787946428571397,0.013113839285714302,0.012276785714285698,0.010323660714285698,0.006975446428571397,0.006696428571428603,0.005301339285714302,0.002790178571428603,0.001953125,0.0016741071428570953,0.0016741071428570953,0.0005580357142856984,0.0005580357142856984,0.0005580357142856984,0.0005580357142856984,0.0002790178571429047]
max_passmark = 3223.49


assert len(correction_coefficients) == len(percentages_higher_than_predicted)


def _get_proportional_value(ref_array, return_values_array, ref_value):
    if ref_value > ref_array[0] or ref_value < ref_array[-1]:
        print("ERROR: ref_value value should be in the interval (", ref_array[-1], ", ", ref_array[0], ")", sep="")
        print("ref_value =", ref_value, "was given.")
        exit(1)
    for i in range(len(ref_array)):
        if ref_value >= ref_array[i]:
            # Just a weighted average between two closest values.
            proportion1 = abs(ref_array[i-1] - ref_value)
            proportion2 = abs(ref_array[i] - ref_value)
            proportion1, proportion2 = proportion1 / (proportion1 + proportion2),  proportion2 / (proportion1 + proportion2)
            proportion1 = 1.0 - proportion1
            proportion2 = 1.0 - proportion2
            return return_values_array[i-1] * proportion1 + return_values_array[i] * proportion2


def get_correction_coefficien_from_probability(probability):
    """Get the correction coefficient required for a target probability of predicting an unfairly longer runtime.

    The target probability needs to be in the interval (1.0, 0.5). 

    Parameters
    ----------
    probability : float
        The probability of predicting a unfairly longer runtime associated to the correction coefficient returned.
    """

    if probability > percentages_higher_than_predicted[0] or probability < percentages_higher_than_predicted[-1]:
        print("ERROR: probability should be in the interval (", percentages_higher_than_predicted[-1], ", ", percentages_higher_than_predicted[0], ")", sep="")
        print("probability =", probability, "was given.")
        exit(1)
    return _get_proportional_value(percentages_higher_than_predicted, correction_coefficients, probability)



def get_probability_from_correction_coefficient(correction_coefficient):
    """Get the probability of predicting an unfairly longer runtime associated to a correction_coefficient.

    The correction coefficient needs to be in the interval (0.00028, 0.5). 

    Parameters
    ----------
    correction_coefficient : float
        The correction coefficient for which we compute the probabilty of predicting an unfairly longer runtime.
    """

    if correction_coefficient > correction_coefficients[0] or correction_coefficient < correction_coefficients[-1]:
        print("ERROR: correction_coefficient should be in the interval (", correction_coefficients[-1], ", ", correction_coefficients[0], ")", sep="")
        print("correction_coefficient =", correction_coefficient, "was given.")
        exit(1)
    return _get_proportional_value(correction_coefficients, percentages_higher_than_predicted, correction_coefficient)


def get_runtime_adjustment_proportion(s1, s2, correction_coefficient):
    """Get the proportion of runtime that machine 2 deserves with respect to machine 1.

    We know the runtime in machine 1, and we want to know what runtime machine 2 deserves with respect to the runtime in machine 1.
    For example, if this function returns 0.5, then machine 2 deserves half of the runtime that machine 1 had. 
    A correction coefficient of 1.0 makes the prediction centered: same probability of predicting a higher or a lower runtime.

    Parameters
    ----------
    s1 : float
        The single thread PassMark CPU score of machine 1.

    s2 : float
        The single thread PassMark CPU score of machine 2.

    correction_coefficient : float
        The correction coefficient.
    """

    return (max_passmark - s1) / (max_passmark - s2) * correction_coefficient


def get_runtime_adjustment_proportion_from_probability(s1, s2, target_probability):
    """Get the proportion of runtime that machine 2 deserves with respect to machine 1, given the probability of predicting a longer runtime in machine 2.

    A probability of 0.5 makes the prediction centered, same probability of predicting a higher or a lower runtime.

    Parameters
    ----------
    s1 : float
        The single thread PassMark CPU score of machine 1.

    s2 : float
        The single thread PassMark CPU score of machine 2.

    target_probability : float
        The probability that an unfairly longer runtime is computed for machine 2.
    """

    if s1 > max_passmark or s2 > max_passmark:
        print("ERROR: Due to the fitting of the prediction of the runtime, machines with Passmark higher than", max_passmark, "are not compatible with the framework.")

    return (max_passmark - s1) / (max_passmark - s2) * get_correction_coefficien_from_probability(target_probability)




def get_equivalent_runtime(s1, s2, t_1, correction_coefficient):
    """Get t_2: the equivalent runtime of t_1 in machine 2.

    The runtime in machine 1 is t_1, and this function returns the runtime machine 2 deserves.
    A correction coefficient of 1.0 makes the prediction centered: same probability of predicting a higher or a lower runtime.

    Parameters
    ----------
    s1 : float
        The single thread PassMark CPU score of machine 1.

    s2 : float
        The single thread PassMark CPU score of machine 2.

    t1 : float
        The runtime in machine 1.

    correction_coefficient : float
        The correction coefficient.
    """

    return (max_passmark - s1) / (max_passmark - s2) * correction_coefficient * t_1




def get_equivalent_runtime_from_probability(target_probability, s1, s2, t_1):
    """Get t_2: the equivalent runtime of t_1 in machine 2.

    The runtime in machine 1 is t_1, and this function returns the runtime machine 2 deserves.
    A probability of 0.5 makes the prediction centered, same probability of predicting a higher or a lower runtime.

    Parameters
    ----------
    s1 : float
        The single thread PassMark CPU score of machine 1.

    s2 : float
        The single thread PassMark CPU score of machine 2.

    t1 : float
        The runtime in machine 1.

    target_probability : float
        The probability that an unfairly longer runtime is computed for machine 2.
    """

    if s1 > max_passmark or s2 > max_passmark:
        print("ERROR: Due to the fitting of the prediction of the runtime, machines with Passmark higher than", max_passmark, "are not compatible with the framework.")

    return (max_passmark - s1) / (max_passmark - s2) * get_correction_coefficien_from_probability(target_probability) * t_1



def usage():
    print("This script can be used to calculate the equivalent runtime. It requires four input arguments:")
    print("1) The desired probability of prediciting a unfairly higher runtime, p_gamma. It must be in the interval (0.003, 0.5)")
    print("2) The PassMark score of machine 1")
    print("3) The PassMark score of machine 2")
    print("4) t_1: The runtime in machine 1")
    print("")
    print("Returns the estimated equivalent runtime")
    print("----")
    print("Example: ")
    print("python equivalent_runtime.py 0.5 1540 1643 23.2")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        usage()
        exit(0)
    print('{:.8}'.format(get_equivalent_runtime_from_probability(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))))
