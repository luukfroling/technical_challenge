# Introduction

Running train.py using my laptop would (according to my calculations) have taken about 3 days. Instead, I will be using google colab to run the notebook files, which I will then download and add in the Notebooks folder of this project. 

This website is a [Myst document](https://mystmd.org/), which allows me to write about some of the changes I made to the `train.py` file in a structured way, combining the outputs of codecells with markdown text. For any figure, the `source : ...` will be displayed in the top right corner. Clicking on this will navigate the page to the source notebook file. 

This project contains the following files: 
- Initial run - No changes made to the code, only for logging purposes. 
- First step - I changed the optimiser to AdamW
- Second step - 



# Initial run

We first run train.py to check the validation loss to beat. 

:::{figure} #loss_first_run
:label: label_loss_first_run
:::

[](#label_loss_first_run) shows the loss as training progresses, settling on `loss=1.753273` after 7 epochs. Looking at the type of outputs this model produces, I noticed the following.  

:::{figure} #token_frequency_first_run
:label: label_token_frequency_first_run
:::

[](#label_token_frequency_first_run) shows for a given input used in the validation sequence, the model incorrectly predicts `<eos>`. For each input, the model would do this. 

:::{figure} #token_frequency_input
:label: label_token_frequency_input
:::

# First step - optimiser 

# Second step - 