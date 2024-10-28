# Policy Aggregation 

Code for the experiments.

### To install the dependencies run 
```
pip install -r pip-requirements.txt  
```

## To run the approaches proposed in the paper run the following commands: 
Start by creating the environment and generating random sample polices. This creates an environment and stores it in exp_results directory. 
```
python -m fact_exp create 5 3 500000
```
Then to reproduce the experiment shown in Figure 1 (a), run the following command: 
```
python -m fact_exp subsets exp_result/{name of generated environment}
```
Then to reproduce the experiment shown in Figure 1 (b), run the following command:
```
python -m fact_exp base exp_result/{name of generated environment}
```