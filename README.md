# L3GD

## Examples in paper

- synthesis dataset
  - Randomly generated $b_t$. 
  - $b_t=10\cos(\frac{\pi}{100})$
  - $b_t=10\cos(\frac{2\pi}{500})$
- MNIST dataset
  - please download `MNISTdata.hdf5` to `./MNIST_examples/dataset`
  - config files are saved in `./MNIST_examples/configs/` 


### Run examples in paper

- Run `bash run_synthesis_examples.sh` for synthesis examples.
  - or equivalently, run the contents in bash file, one by one.
  - the figures are saved in `./synthetic_examples/assets/`
- Run `bash run_MNIST_examples.sh` for MNIST examples.
  - or equivalently, run the contents in bash file, one by one.
  - the figures are saved in `./MNIST_examples/assets/`
  - the models are savedin `./MNIST_examples/models/`
  - the loss function records are saved in ./MNIST_examples/assets/
- Run `cd ./Cifar10_examples`, then `bash run_MNIST_examples.sh`
  - find the test accuracies in the files end with **.log** under the `logs` folder.

## Notes

- Install the libraries listed in `requirements.txt` by running `pip install -r requirements.txt`

## Published paper
The code is the implementation of the following paper.
```
@ARTICLE{10507162,
  author={Qu, Zhihai and Li, Xiuxian and Li, Li and Yi, Xinlei},
  journal={IEEE Transactions on Signal Processing}, 
  title={Online Optimization Under Randomly Corrupted Attacks}, 
  year={2024},
  volume={72},
  number={},
  pages={2160-2172},
  keywords={Optimization;Signal processing algorithms;Heuristic algorithms;Machine learning algorithms;Robustness;Noise;Security;Online optimization;corrupted gradients;resilient online optimization algorithm},
  doi={10.1109/TSP.2024.3392361}
}
```
