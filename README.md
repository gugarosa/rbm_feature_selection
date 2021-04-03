# Binary Masking Feature Selection with Restricted Boltzmann Machines

*This repository holds all the necessary code to run the very-same experiments described in the paper "Binary Masking Feature Selection with Restricted Boltzmann Machines".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure

 * `core`
   * `fsrbm`: Feature Selection Restricted Boltzmann Machine;
   * `rbm`: Restricted Boltzmann Machine;
 * `outputs`: Folder for saving the output files, such as `.pth` and `.txt`;
 * `utils`
   * `loader.py`: Loads, transforms and splits the used datasets.
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

### Data configuration

In order to run the experiments, you can use `torchvision` to load pre-implemented datasets.

---

## Usage

### Model Training

Initially, one needs to train the desired model, i.e., FSRBM or RBM. To accomplish such a procedure, both models can be trained using the following script:

```Python
python model_training.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### Model Reconstruction

After training the desired models, one can use its pre-trained files and reconstruct samples from the test set. Note that if a FSRBM has been trained, there will be an extra output file, which stands for the binary mask that can be used to mask the dataset. Thus, please use the following script to accomplish such a procedure:

```Python
python model_reconstruction.py -h
```

*Again, the `-h` will invoke the script helper. Note that we are outputting the MSE metric, as well as an image from the first reconstructed tensor. Feel free to change accordingly your needs.*

### Model Batch Reconstruction (Optional)

Instead of reconstructing model per model, one can perform a batch reconstruction along the saved models. Such a procedure is useful is there has been snapshots throughout the training epochs, i.e., models were saved every 10 epochs during the 50 training epochs. Please, use the following script:

```Python
python model_batch_reconstruction.py -h
```

*Note that this script should be called only with the files' path stem, i.e., without its numbering. The script will find all the correlated files and will produce a list holding the output metrics.*

### Analyze Metrics using Convergence Plots (Optional)

Additionally, it is possible to construct a convergence plot and compare the metrics amongst the distinct evaluated models. Essentially, this script can be used within any type of metric, just need to be feed with the corresponding list of values:

```Python
python plot_convergence_analysis.py -h
```

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br and mateus.roder@unesp.br.

---
