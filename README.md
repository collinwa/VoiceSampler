# Voice Sampler

A machine learning based model developed to sample voices out of pop songs. Current implementation uses a U-Net with gated recurrent units at the highest depth of convolution


## Current list of todos:

### Data Processing 
* Develop an training / test data split for separate evaluation
* Cut out the first 5-6 seconds to remove introduction from vocal samples
* Find a better way to mix samples, besides taking the average (?)
* Parallelize data acquisition using torch.DataLoader so that the GPUs can be utilized more fully
    * Experiment with different values for num_workers to see how things are getting accelerated
    * Experiment with different queue sizes to determine how to get the greatest GPU utilization
* Parallelize training across multiple GPUs using torch.DataParallel

### Model Development
* ~~ Finish writing the U-Net ~~
* ~~ Implement recurrent units ~~
* Implement dropout to prevent overfitting
* Use [Ax](https://ax.dev/docs/api.html) to tune model hyperparameters and fix convergence issues
* Integrate a perceptual loss function into the model, instead of SSE
* Implement GRU skip connections

### Feature Engineering
* Mel spectrogram (?) (don't really know what this is...)
* Find bulk metrics or other data that may be useful for network to sample out the voices
* Unsupervised / supervised pre-training on current voice data sets using CNNs
* Generative modeling using a pre-built wave-net model to overlay voices

### Data Acquisition
* ~~ Get royalty free music ~~
* ~~ Get a capella voices ~~
* Get more a capella voices (optional)
* Get more instrumentals (optional)