- 

# Introduction

Running train.py using my laptop would (according to my calculations) have taken about 3 days. Instead, I will be using google colab to run the notebook files, which I then download and place in the Notebooks folder of this project. All notebooks can be accessed under 'supporting documents'.

This website is build using [Myst](https://mystmd.org/), which lets me write about changes I made to `train.py` in a structured way, combining code-cell outputs with markdown text. For any figure, a `source : ...` button will appear in the top right corner, clicking it will jump directly to the original notebook file.

The goal of this project is:

> Your task is to analyze and improve the training run of a small GPT-2 style model. The code is designed to train on a Hacker News headline dataset. Your goal is to minimize the validation loss as low as possible within 7 epochs. The baseline to beat is a validation loss of 1.754.

Where I aim to:
- Minimise the validation loss. 
- Understand how well the model performs. 



# Initial run

The first step is to run the original train.py file as provided on the GitHub page.

:::{figure} #loss_first_run
:label: label_loss_first_run
:::

[](#label_loss_first_run) shows how the loss decreases as training progresses, converging to `loss=1.753273` after 7 epochs. We can also have a look at an example of the output provided by the model. For this, I'll look at `batch = 0` and `time = 10` within the `evaluate_visualise` function. The decoded input to the model is:
> `Motorola Xoom commercial reminiscent of Apple's`

The model's top-10 predicted next tokens are shown in [](#label_token_frequency_first_run).

:::{figure} #token_frequency_first_run
:label: label_token_frequency_first_run
:::

As seen in the figure, the model incorrectly predicts `<eos>`. In fact, the model predicts this token for every input I checked.

:::{figure} #token_frequency_input
:label: label_token_frequency_input
:::

[](#label_token_frequency_input) displays the most frequently used words in the input data. This distribution matches the model's predictions in [](#label_token_frequency_first_run), which tells us the model is not making the predictions based on context, but rather based on statistics (clever, but now what we are looking for). The aim now is not only to improve the validation loss, but also to adjust the model so it produces more sensible next-token predictions rather than defaulting to `<eos>`.

# Step one - AdamW Optimiser 

The original model uses an SGD optimiser, which is not ideal for training attention-based architectures. I switched to AdamW instead, which is the optimiser typically used for modern transformer and LLM training.

```python
# Add different optimiser, which is better suited for transformer models
def get_optimizer(model, args):
    # Add AdamW as an optimiser
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
```

I added a separate `get_optimizer` function so the change is easy to spot at the top of the code cell. I also reduced the learning rate and added explicit beta values for AdamW: 

```python
n_layer: int = 6
n_head: int = 8
d_model: int = 512
dropout: float = 0.1
lr = 5e-4                   #lr: float = 6e-3, 
weight_decay: float = 0.0
evals_per_epoch: int = 3
betas = (0.9, 0.95)         # Added betas for AdamW
```

Running the model with these changes resulted in a validation loss of 1.32, a noticeable improvement over the original program. More importantly, the model produces far more reasonable next-token suggestions. For example, when given the input: 

> Motorola Xoom commercial reminiscent of Apple's

the predictions look much better, as shown in [](#label_step_one_words)

:::{figure} #step_one_words
:label: label_step_one_words
:::

Here, the top prediction is “new”, which fits naturally after the given fragment.


# Step two - Positional context/increased vocab

I noticed that each training sample’s context often contains multiple titles in a row, but earlier titles offer no useful information for predicting later ones. For example, the first batch may include an input such as: 

> `Motorola Xoom commercial reminiscent of Apple's infamous jab at IBM<eos> OpenSSH Key Shielding<eos> Contextual Word Representations: A Contextual Introduction (2020)<eos> Windows 8 Should Be Free<eos> GNOME Developers `

The next word after “GNOME Developers” has nothing to do with the earlier titles. Positional information should help the model focus on the current sentence and ignore older ones. I have added [Rotary Position Embedding (RoPE) ](https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/), wwhich I was unfamiliar with before the project but found straightforward to integrate. 

In addition, I increased the tokenizer vocab size and added RSMNorm. Tasks with many unique tokens (technical words, product names, company names, and acronyms) tend to benefit from a larger vocabulary because it reduces over-fragmentation during BPE tokenization. 

With these changes combined, I was able to bring the validation loss down to 1.25. After training, the output distribution for the same input as [](#label_token_frequency_first_run) is shown below. 

:::{figure} #output_step_two_token_10
:label: label_output_step_two_token_10
:::

Given the sentence so far (`Motorola Xoom commercial reminiscent of Apple's`), the top two predictions (“new” and “first”) are both reasonable continuations.

Another interesting case is to look at a longer string. By setting $b=0 and t=80$ in the evaluate_visualise function, we get
> `Motorola Xoom commercial reminiscent of Apple's infamous jab at IBM<eos> OpenSSH Key Shielding<eos> Contextual Word Representations: A Contextual Introduction (2020)<eos> Windows 8 Should Be Free<eos> GNOME Developers Lay Out Plans for GNOME OS<eos> HTML/CSS/JS Art Hacks<eos> Major Electrical Outtage at The Planet Data Center<eos> South Africa’s Umhlanga Rocks may attract investors<eos>`

At this point, the model must decide how to start a brand-new sentence.

:::{figure} #token_predictions_two
:label: label_token_predictions_two
:::

[](#label_token_predictions_two) shows the predicted probability distribution. For the start of a fresh sentence, any of the top-ten tokens would make sense. 

This example also highlights the biggest issue I’ve encountered so far: no amount of tuning can help the model guess the next word (“Snapchat”) without additional context. The model has no information about the underlying article or subject matter, only the previous titles. It simply doesn’t have the necessary cues.


# Concluding thoughts 

With a final validation loss of 1.25, further improvements are likely to come from providing more context rather than continuing to tune the existing architecture. One promising direction would be to include the article text itself, allowing the model to extract keywords (e.g., company names, locations, products) and general themes before predicting the title.

This additional information would address the core limitation revealed in the examples: the model often lacks the semantic clues needed to make the “correct” next-token prediction, because that information is nowhere in its current input. Adding this additional context will most likely improve the loss significantly more than small optimisations inside the current pipeline. 

Unfortunately, the rules of the challenge don’t allow adding the article text or modifying the dataset, so I couldn’t explore this direction here.