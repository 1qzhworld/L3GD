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
