Pytorch is required for the code run `bash run_Cifar10_examples.sh` for the reproduction.



The table in the rebuttal is the average of the results after 200 epoches by running `bash run_Cifar10_examples.sh`.

| Attack Prob | 0.35  | 0.37  | 0.39  | 0.41  | 0.43  | 0.45  |
| ----------- | ----- | ----- | ----- | ----- | ----- | ----- |
| OMGD        | 10.0% | 10.0% | 10.0% | 10.0% | 10.0% | 10.0% |
| ONGD        | 84.5% | 82.5% | 82.0% | 77.1% | 71.8% | 57.4% |
| L3GD        | 83.8% | 83.8% | 84.8% | 82.1% | 79.8% | 65.4% |

The setting of the experiments are collected in the files in folder configs, where L3GD set the $K_t$ for the inner loop as `3`. 
