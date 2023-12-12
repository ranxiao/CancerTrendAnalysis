def ta_l1_tf(*args):
    """
    Apply l1 trend filtering to generate piecewise linear trends in time series signal
    When provided with signal time and sampling frequency, the function integrate time awareness to handle irregular time intervals in the signal

    Parameters:
        signal (float): The number for which factorial is to be calculated.
        lmd(float): the regularization parameter lambda that control the level of l1 regularization
        time(float) (optional): time of signal in seconds

    Returns:
        float: L1 trend filtered data.

    Author: Ran Xiao, Emory University, ran.xiao@emory.edu 06/07/2023
    """
    if len(args) < 2:
        raise TypeError("At least two arguments are required, including signal and regularization parameter lambda.")

    try:
        import numpy as np
        import cvxpy as cp
        import scipy as scipy
    except ImportError:
        raise ImportError("The 'numpy','cvxpy' and 'scipy' packages are required to calculate the factorial.")
    # supress all warnings
    import warnings
    warnings.filterwarnings("ignore")

    signal =args[0]
    lmd = args[1]
    # # Form second difference matrix.
    n = signal.size
    e = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((e, -2 * e, e)), range(3), n - 2, n)

    if len(args) == 3:
        time = args[2]

        # Compute time differences
        try:
            delta_t = np.diff(time)
            delta_t_max = np.max(delta_t)
            delta_t_norm = delta_t / delta_t_max  # normalize by maximum delta time in the signal
            delta_t_norm_inv = 1 / delta_t_norm
        except ValueError:
            raise ValueError("Issue with signal time, please check for duplication of time")

        dense_matrix = D.toarray()

        for i in range(dense_matrix.shape[0]):
            dense_matrix[i, i] = delta_t_norm_inv[i]
            dense_matrix[i, i + 1] = - delta_t_norm_inv[i] - delta_t_norm_inv[i + 1]
            dense_matrix[i, i + 2] = delta_t_norm_inv[i + 1]

        D = scipy.sparse.csr_matrix(dense_matrix)

    x = cp.Variable(shape=n)

    obj = cp.Minimize(0.5 * cp.sum_squares(signal - x)
                      + lmd * cp.norm(D * x, 1))
    prob = cp.Problem(obj)

    # List of solvers to try in order, ECOS, SCS, and OSQP
    solvers = [cp.ECOS, cp.SCS, cp.OSQP]
    #ECOS is best for small to medium-sized SOCPs, where simplicity and embedding are important.
    #SCS excels in handling large, sparse problems, particularly those involving SDPs and complex cone structures.
    #OSQP is specialized for quadratic programs and shines in situations where problems can be formulated as such, offering high efficiency and robustness.

    # Try to solve the problem using each solver in turn
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status == cp.OPTIMAL:
                break  # Break out of the loop if a solution is found
        except Exception as e:
            print(f"An error occurred with solver {solver}: {str(e)}")
            continue  # Continue to the next solver if an error occurs

    # Check if the problem was solved optimally
    if prob.status != cp.OPTIMAL:
        raise Exception("None of the solvers converged!")

    # Calculate the second difference (using the D matrix)
    second_diff = D @ x.value

    # compare kink with a threshold, e.g. 1e-8, and assign 1 to kink if kink > threshold, 0 otherwise
    kink = np.where(abs(second_diff) > 1e-8, 1, 0)

    # calculate number of segments
    segment_num = np.sum(kink) + 1

    # # plot x.value, signal, along time
    # import matplotlib.pyplot as plt
    # plt.plot(time, signal, 'k.', label='signal')
    # # plot x.value along time by dots representing point and lines connecting points
    # plt.plot(time, x.value, 'r.', label='l1tf')
    # plt.legend()
    # plt.show()


    # return x.value, D_ta, D_reg
    return x.value, kink, segment_num, delta_t

def l1tf_lambdamax(y, t=None):
    """
    Returns an upperbound of lambda. With a regularization parameter value over lambda_max,
    l1tf returns the best affine fit for y.

    Parameters:
        y (np.array): n-vector; original signal
        t (np.array, optional): time points of signal in seconds

    Returns:
        lambdamax (float): maximum value of lambda in useful range
    """
    try:
        import numpy as np
        from scipy.sparse.linalg import spsolve
        from numpy.linalg import norm
        from scipy.sparse import diags
        import scipy as scipy
    except ImportError:
        raise ImportError("The 'numpy' and 'scipy' packages are required to calculate the factorial.")

    n = len(y)
    e = np.ones((1, n))
    D_reg = scipy.sparse.spdiags(np.vstack((e, -2 * e, e)), range(3), n - 2, n)
    if t is None:  # assume regular time intervals
        e = np.ones((1, n))
        D = D_reg
    else:  # irregular time intervals
        dense_matrix = D_reg.toarray()
        delta_t = np.diff(t)
        delta_t_max = np.max(delta_t)
        delta_t_norm = delta_t / delta_t_max  # normalize by maximum value
        delta_t_norm_inv = 1 / delta_t_norm

        for i in range(dense_matrix.shape[0]):
            dense_matrix[i, i] = delta_t_norm_inv[i]
            dense_matrix[i, i + 1] = - delta_t_norm_inv[i] - delta_t_norm_inv[i + 1]
            dense_matrix[i, i + 2] = delta_t_norm_inv[i + 1]

        D = scipy.sparse.csr_matrix(dense_matrix)

    lambdamax = norm(spsolve(D.dot(D.transpose()), D.dot(y)), np.inf)

    return lambdamax

def getTrendFeatures(x, lmd,t,flag_plt=0):
    """
    Returns the trend features for each slope of the signal x.
    """

    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        raise ImportError("The 'numpy' and 'pandas' packages are required to calculate the factorial.")

    trends, kink, segment_num,delta_t = ta_l1_tf(x, lmd,t)
    if flag_plt == 1:
        # plot out x and trends with t
        import matplotlib.pyplot as plt
        plt.plot(t,x,'k.',label='signal')
        plt.plot(t,trends,'r-',label='l1tf')
        plt.show()

    kink_index = list(np.nonzero(kink)[0])
    kink_value = trends[np.nonzero(kink)[0]+1]

    # generate a pd dataframe to store the trend features for each slope
    trend_features = pd.DataFrame(columns=['slope','duration','total_dur','start_v','end_v','sign','lmd'])
    if len(kink_index) > 0:
        for i in range(len(kink_index)+1): # n kinks indicate n+1 trend
            if i == 0:# the first trend
                temp_slope = (trends[kink_index[i]+1] - trends[kink_index[i]])/delta_t[kink_index[i]] # delta_t share the same indices with kink
                temp_dur = sum(delta_t[:kink_index[i]+1])
                temp_start_v = trends[0]
                temp_end_v = kink_value[i]
                temp_sign = 1 if temp_slope > 0 else -1
            elif i == len(kink_index):# the last trend
                temp_slope = (trends[kink_index[-1]+2]-trends[kink_index[-1]+1])/delta_t[kink_index[-1]+1] # delta_t share the same indices with kink
                temp_dur = sum(delta_t[kink_index[-1]+1:])
                temp_start_v = trends[kink_index[-1]+1]
                temp_end_v = trends[-1]
                temp_sign = 1 if temp_slope > 0 else -1
            else:# the middle trends
                temp_slope = (trends[kink_index[i]+1] - trends[kink_index[i]])/delta_t[kink_index[i]] # delta_t share the same indices with kink
                temp_dur = sum(delta_t[kink_index[i-1]+1:kink_index[i]+1])
                temp_start_v = trends[kink_index[i-1]+1]# trend value index is one more than kink index
                temp_end_v = trends[kink_index[i]+1]
                temp_sign = 1 if temp_slope > 0 else -1

            slope_totalDuration = sum(delta_t)
            trend_features.loc[i] = [temp_slope,temp_dur,slope_totalDuration,temp_start_v,temp_end_v,temp_sign,lmd]
    else:
        temp_slope = (trends[-1] - trends[0])/sum(delta_t)
        temp_dur = sum(delta_t)
        temp_start_v = trends[0]
        temp_end_v = trends[-1]
        temp_sign = 1 if temp_slope > 0 else -1

        slope_totalDuration = sum(delta_t)
        trend_features.loc[0] = [temp_slope,temp_dur,slope_totalDuration,temp_start_v,temp_end_v,temp_sign,lmd]
    return trend_features, trends

def genFeatureTable(trendFeatures):
    # group by lmd
    trendFeatures_grouped = trendFeatures.groupby(['lmd'])

    import pandas as pd
    # now we have 31 features
    feature_table = pd.DataFrame(columns=['lmd', 'segment_num',
                                          'slope_pos_max', 'slope_pos_min', 'slope_pos_median', 'slope_pos_mean',
                                          'slope_neg_max', 'slope_neg_min', 'slope_neg_median', 'slope_neg_mean',
                                          'slope_pos_percent', 'slope_pos_duration_percent',
                                          'slope_neg_percent', 'slope_neg_duration_percent',
                                          'start_v_pos_max', 'start_v_pos_min', 'start_v_pos_median',
                                          'start_v_pos_mean',
                                          'start_v_neg_max', 'start_v_neg_min', 'start_v_neg_median',
                                          'start_v_neg_mean',
                                          'end_v_pos_max', 'end_v_pos_min', 'end_v_pos_median', 'end_v_pos_mean',
                                          'end_v_neg_max', 'end_v_neg_min', 'end_v_neg_median', 'end_v_neg_mean'])
    # calculate the feature_table for each group
    for name, group in trendFeatures_grouped:
        # get the number of segments in the window
        segment_num = len(group)
        # get the number of positive and negative slopes
        slope_pos_num = len(group[group['sign'] == 1])
        slope_neg_num = len(group[group['sign'] == -1])
        # get the percentage of positive and negative slopes
        slope_pos_percent = len(group[group['sign'] == 1]) / segment_num
        slope_neg_percent = len(group[group['sign'] == -1]) / segment_num

        # get the percentage of positive and negative slope duration
        slope_pos_duration_percent = sum(group[group['sign'] == 1]['duration']) / sum(
            group['duration']) if slope_pos_num > 0 else 0
        slope_neg_duration_percent = sum(group[group['sign'] == -1]['duration']) / sum(
            group['duration']) if slope_neg_num > 0 else 0

        # get the max, min, median, mean of slope_pos, slope_neg, start_v_pos, start_v_neg, end_v_pos, end_v_neg
        slope_pos_max = group[group['sign'] == 1]['slope'].max() if slope_pos_num > 0 else 0
        slope_pos_min = group[group['sign'] == 1]['slope'].min() if slope_pos_num > 0 else 0
        slope_pos_median = group[group['sign'] == 1]['slope'].median() if slope_pos_num > 0 else 0
        slope_pos_mean = group[group['sign'] == 1]['slope'].mean() if slope_pos_num > 0 else 0

        slope_neg_max = abs(group[group['sign'] == -1]['slope']).max() if slope_neg_num > 0 else 0
        slope_neg_min = abs(group[group['sign'] == -1]['slope']).min() if slope_neg_num > 0 else 0
        slope_neg_median = abs(group[group['sign'] == -1]['slope']).median() if slope_neg_num > 0 else 0
        slope_neg_mean = abs(group[group['sign'] == -1]['slope']).mean() if slope_neg_num > 0 else 0

        start_v_pos_max = group[group['sign'] == 1]['start_v'].max() if slope_pos_num > 0 else 0
        start_v_pos_min = group[group['sign'] == 1]['start_v'].min() if slope_pos_num > 0 else 0
        start_v_pos_median = group[group['sign'] == 1]['start_v'].median() if slope_pos_num > 0 else 0
        start_v_pos_mean = group[group['sign'] == 1]['start_v'].mean() if slope_pos_num > 0 else 0

        start_v_neg_max = group[group['sign'] == -1]['start_v'].max() if slope_neg_num > 0 else 0
        start_v_neg_min = group[group['sign'] == -1]['start_v'].min() if slope_neg_num > 0 else 0
        start_v_neg_median = group[group['sign'] == -1]['start_v'].median() if slope_neg_num > 0 else 0
        start_v_neg_mean = group[group['sign'] == -1]['start_v'].mean() if slope_neg_num > 0 else 0

        end_v_pos_max = group[group['sign'] == 1]['end_v'].max() if slope_pos_num > 0 else 0
        end_v_pos_min = group[group['sign'] == 1]['end_v'].min() if slope_pos_num > 0 else 0
        end_v_pos_median = group[group['sign'] == 1]['end_v'].median() if slope_pos_num > 0 else 0
        end_v_pos_mean = group[group['sign'] == 1]['end_v'].mean() if slope_pos_num > 0 else 0

        end_v_neg_max = group[group['sign'] == -1]['end_v'].max() if slope_neg_num > 0 else 0
        end_v_neg_min = group[group['sign'] == -1]['end_v'].min() if slope_neg_num > 0 else 0
        end_v_neg_median = group[group['sign'] == -1]['end_v'].median() if slope_neg_num > 0 else 0
        end_v_neg_mean = group[group['sign'] == -1]['end_v'].mean() if slope_neg_num > 0 else 0

        # append the feature_table
        feature_table = pd.concat(
            [feature_table, pd.DataFrame({'lmd': [name], 'segment_num': [segment_num],
                                          'slope_pos_percent': [slope_pos_percent],
                                          'slope_neg_percent': [slope_neg_percent],
                                          'slope_pos_duration_percent': [slope_pos_duration_percent],
                                          'slope_neg_duration_percent': [slope_neg_duration_percent],
                                          'slope_pos_max': [slope_pos_max], 'slope_pos_min': [slope_pos_min],
                                          'slope_pos_median': [slope_pos_median], 'slope_pos_mean': [slope_pos_mean],
                                          'slope_neg_max': [slope_neg_max], 'slope_neg_min': [slope_neg_min],
                                          'slope_neg_median': [slope_neg_median], 'slope_neg_mean': [slope_neg_mean],
                                          'start_v_pos_max': [start_v_pos_max], 'start_v_pos_min': [start_v_pos_min],
                                          'start_v_pos_median': [start_v_pos_median],
                                          'start_v_pos_mean': [start_v_pos_mean],
                                          'start_v_neg_max': [start_v_neg_max], 'start_v_neg_min': [start_v_neg_min],
                                          'start_v_neg_median': [start_v_neg_median],
                                          'start_v_neg_mean': [start_v_neg_mean],
                                          'end_v_pos_max': [end_v_pos_max], 'end_v_pos_min': [end_v_pos_min],
                                          'end_v_pos_median': [end_v_pos_median], 'end_v_pos_mean': [end_v_pos_mean],
                                          'end_v_neg_max': [end_v_neg_max], 'end_v_neg_min': [end_v_neg_min],
                                          'end_v_neg_median': [end_v_neg_median], 'end_v_neg_mean': [end_v_neg_mean]})], ignore_index=True)
    return feature_table
