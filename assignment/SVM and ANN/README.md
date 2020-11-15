
# Brief Description:    
This directory contains the following files:    
- `Assignment 3B.pdf`: Description of problem statement  
- `ann.py`: Contains the functions related to the MLP Classifier
- `biodeg.csv`: Contains the data used for training both MLP and Binary SVM Classifier  
- `comparison_plots.png`: The plots of **learning rate vs accuracy for each model** (in top row)  
    and **model vs  accuracy for each learning rate** (in bottom row)  
- `main.py`: Contains the solution to problems provided in the `Assignment 3B.pdf`        
- `requirements.txt`: Contains all the necessary dependencies and their versions     
- `simulations.txt`: Sample simulation output on entire data  
- `svm.py`: Contains the functions related to the Binary SVM Classifier  
- `utils.py`: Contains all the helper functions used by the above files (if any) 

# Directions to use the code  
1. Download this directory into your local machine

2. Copy the file `biodeg.csv` to the directory where the code resides

3. Ensure all the necessary dependencies with required version and latest version of Python3 are available (verify with `requirements.txt`)  <br>
 `pip3 install -r requirements.txt`

4. Run specific functions with the aid of `main.py` <br>

# For giving the **maxc** parameter (the maximum value of C to be checked (on log-scale) would be 10<sup>maxc</sup>)
- Using the default maxc = 4  
`python3 main.py`  

- Giving input integer maxc (say 2 i.e., maximum C value to be checked would be 10<sup>2</sup>) -- integer should be greater than or equal to 0  
`python3 main.py --maxc 2`

- For more help regarding the arguments  
`python3 main.py --help`
