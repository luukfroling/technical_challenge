# Introduction

Running train.py using my laptop would (according to my calculations) have taken about 3 days. Instead, I will be using google colab to run the notebook files, which I will then download and add in the Notebooks folder of this project. All notebooks can be accessed under 'supporting documents'.

This website is a [Myst document](https://mystmd.org/), which allows me to write about some of the changes I made to the `train.py` file in a structured way, combining the outputs of codecells with markdown text. For any figure, the `source : ...` will be displayed in the top right corner. Clicking on this will navigate the page to the source notebook file. 

This project contains the following files: 
- Initial run - No changes made to the code, only for logging purposes. 
- First step - I changed the optimiser to AdamW
- Second step - 

The goal is to 
- Check what sort of inputs the model receives, and the type of outputs we are looking for.
- Minimise the validation loss. 



# Initial run

First step is to run the orignial `trai.py` file as provided on the GitHub page.

:::{figure} #loss_first_run
:label: label_loss_first_run
:::

[](#label_loss_first_run) shows the loss as training progresses, converging towards `loss=1.753273` after 7 epochs. We can also have a look at an example of the output provided by the model. For this, we will look at `batch = 0` and `time = 10`. The decoded input to the model equals `Motorola Xoom commercial reminiscent of Apple's`, and the output tokens are displayed in [](#label_token_frequency_first_run) (top 10 probabilities).

:::{figure} #token_frequency_first_run
:label: label_token_frequency_first_run
:::

As seen in the figure, the model incorrectly predicts `<eos>`. For each input, the model would do this. After looking at the frequency of words used in the data, as displayed below in [](#token_frequency_input), we can see this matches.

:::{figure} #token_frequency_input
:label: label_token_frequency_input
:::

so the goal is not only to improve on the validation loss, but also to see 

# First step - optimiser 

first, the current model uses 

# Second step - 