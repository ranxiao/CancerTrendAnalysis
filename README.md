# CancerTrendAnalysis
L1 trend analysis code and samples for cancer lab values

# Env setup using conda:
conda create --name cvxpy_env
conda activate cvxpy_env
conda install -c conda-forge cvxpy
conda install -c conda-forge pandas
conda install -c conda-forge matplotlib

# output:
1. images of L1 trend filtering results,
2. a pickle file with the file names and corresponding trend features
3. CSV files that record file names that have abnormal trend analysis results, including files with values smaller than 3 time points, files with all entries the same numeric values and files with no optimization solutions.
   
