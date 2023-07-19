# happyleaf

Happyleaf is a deep learning classifier for plant diseases based on the dataset
from the 2019 work [Data for: Identification of Plant Leaf Diseases Using a
9-layer Deep Convolutional Neural Network](https://data.mendeley.com/datasets/tywbtsjrjv/1)
by Arun Pandian J and Geetharamani Gopal (CC0 1.0 license). We use the augmented
dataset here. This is an educational project built using
[JAX](https://github.com/google/jax) and does not make use of neural network or
optimizer libraries built on top of this machine learning framework.

## Architecture and training

The model is based on a two level transformer architecture. The images are read
in as (64, 64, 3) tensors (so as 64 times 64 RGB images) and mapped via a per
pixel linear layer to a (64, 64, 16) tensor in order to increase the latent
dimension to 16. Our goal is to predict the one-hot encoding of the category the
image belongs to, which is a vector of shape (39).

We subdivide the (64, 64, 16) image into 64 patches of shape (8, 8, 16). To each
of these patches we apply the same sequence of transformer blocks, so for each
of the patches, there is communication (via the attention mechanism) in between
the 64 latent vectors corresponding to each patch. We then concatenate the
information of each patch into one big latent vector which we project down to
the latent dimension again. This leaves us with a tensor of shape (8, 8, 16) and
we now define a transformer on these 64 latent vectors just as before, except
that now we only run one of these transformers (since we do not subdivide into
patches beforehand). Finally, we consider two perceptron layers to get logits
which we compare to the one-hot encoding of the output categories with
cross-entropy as a loss function.

We are making use of dropout and layer norms to stabilize the training and use
the AdamW optimizer together with gradient clipping. So far, the model has only
been trained for 50,000 iterations (with a batch size 64), corresponding roughly
to 65 epochs.

## Setup and execution

First, make sure that you have [NumPy](https://numpy.org/),
[Matplotlib](https://numpy.org/) and [OpenCV](https://opencv.org/) installed. If
you are using `pip` you can do so by running

```bash
pip install numpy matplotlib opencv-python
```

This project is based on the JAX framework which you will also need to have
installed. For installation instructions, see
[JAX](https://github.com/google/jax). Next, download the augmented dataset from
the aforementioned website, rename the unzipped folder
`Plant_leave_diseases_dataset_with_augmentation` to `data` and create the file
`numpydata.pkl` using `dataload.ipynb`. The training and testing code of the
model can be found in the Jupyter notebook `main.ipynb` and the checkpoints are
in the `checkpoints` directory.

## License

This project is licensed under the MIT License.

