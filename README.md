Download Link:https://programming.engineering/product/csc413-2516-assignment-3-solution/


# CSC413-2516-Assignment-3-Solution
CSC413/2516 Assignment 3 Solution
You will be completing this assignment with the aid of large language models (LLMs) such as ChatGPT, text-davinci-003, or code-davinci-002. To alleviate the unnecessary steps related to generating results and screenshotting, we have provided the GPT-generated solution with minimum prompting effort in ChatGPT: clauses. The goal is to help you (i) develop a solid understanding of the course materials, and (ii) gain some insight in problem-solving with LLMs. Think of this as analogous to (i) understanding the rules of addition and multiplication, and (ii) learning how to use a calculator. Note that LLMs may not be a reliable “calculator” (yet) — as you will see, GPT-like models can generate incorrect and contradicting answers. It is, therefore important that you have a good grasp of the lecture materials, so that you can evaluate the correctness of the model output, and also prompt the model toward the correct solution.

Prompt engineering. In this assignment, we ask that you try to (i) solve the problems yourself, and (ii) use LLMs

to solve a selected subset of them. You will “guide” the LLMs toward desired outcomes by typing text prompts into the models. There are a number of different ways to prompt an LLM, including direct copy-pasting LATEX strings of a written question, copying function docstrings, or interactively editing the previously generated results. Prompting offers a natural and intuitive interface for humans to interact with and use LLMs. However, LLM-generated solutions depend significantly on the quality of the prompt used to steer the model, and most effective prompts come from a deep understanding of the task. You can decide how much time you want to spend as a university student vs. a prompt engineer, but we’d say it’s probably not a good idea to use more than 25% of your time on prompting LLMs. See Best Practices below for the basics of prompt engineering.

What are LLMs good for? We have divided the assignment problems into the following categories, based on our judgment of how difficult it is to obtain the correct answer using LLMs.

[Type 1] LLMs can produce almost correct answers from rather straightforward prompts, e.g., minor modification of the problem statement.

[Type 2] LLMs can produce partially correct and useful answers, but you may have to use a more sophisticated prompt (e.g., break down the problem into smaller pieces, then ask a sequence of questions), and also generate multiple times and pick the most reasonable output.

[Type 3] LLMs usually do not give the correct answer unless you try hard. This may include problems with involved mathematical reasoning or numerical computation (many GPT models do not have a built-in calculator).

[Type 4] LLMs are not suitable for the problem (e.g., graph/figure-related questions).

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

Written Assignment

What you have to submit for this part

See the top of this handout for submission directions. Here are the requirements.

The zero point questions (in black below) will not be graded, but you are more than welcome to include your answers for these as well in the submission.

For (nonzero-point) questions labeled [Type 1] [Type 2] you need to submit your own solution. Your own solution can be a copy-paste of the LLM output (if you verify that it is correct), but make sure you cite the model properly.

For (nonzero-point) questions in [Type 3] [Type 4] you only need to submit your own written solution, but we encourage you to experiment with LLMs on some of them.

For reference, here is everything you need to hand in for the first half of the PDF report a3-writeup.pdf.

Problem 1: 1.1.2[Type 2] , 1.2.2[Type 1]

Problem 2: 2.1.1[Type 1] , 2.1.2[Type 4] , 2.2[Type 4]

Robustness and Regularization

Adversarial examples plague many machine learning models, and their existence makes the adoption of ML for high-stakes applications undergo increasingly more regulatory scrutiny. The simplest way to generate an adversarial examples is using the untargeted fast gradient sign method (FGSM) from Goodfellow et al. [2014]:

x′ ← x + ϵ sgn(∇xL(f(x ; w), y))

where x ∈ Rd is some training example we want to perturb, y is the label for that example, and ϵ is a positive scalar chosen to be small enough such that the ground truth class of x′ is the same as that of x according to human perception, yet large enough such that our classifier f misclassifies x′ while correctly classifying x. Read about how the sgn() function works here (https://en.wikipedia.org/wiki/Sign_function).

Note that we are taking the gradient of L(f(x ; w), y) with respect to the input x instead of the weights w, and that we are adding this gradient rather than subtracting it since the goal here is to increase the loss on x′.

For the rest of the question, we assume we are dealing with a binary linear classifier that outputs a scalar logit as follows:

f(x ; w) = w⊤x,

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

where w ∈ Rd where d is dimension of the input x, so f : Rd → R. For the remainder of the ques-tion, we ignore the loss function, and simply try to reduce the output predicted by the classifier f.

To simplify our analysis, assume that the linear classifier outputs a positive logit on the input x, w⊤x > 0. The attack is performed on the outputted logits directly to change the model’s prediction from positive to negative. The attack now becomes:

x′ ← x − ϵ sgn(∇xf(x ; w)),

where we are trying to decrease the outputted logit.

1.1 Adversarial Examples

1.1.1 Bounding FGSM [0pt]

To understand why the sgn() function is used, compute the ℓ∞ norm of δ1 = ϵ sgn(∇xf(x ; w)) and δ2 = ϵ ∇xf(x ; w). Under what conditions does ||δ1||∞ = ||δ2||∞? What guarantee does the sgn() function give us on the ℓ∞ norm of the perturbation?

ChatGPT:

The ℓ∞ norm of a vector δ is defined as ||δ||∞ = max i|[δ]i|, where [δ]i denotes the ith entry of the vector. Using this definition, we can compute the norms of δ1 and δ2 as follows:

||δ1||∞ = max |[ϵ sgn(∇xf(x ; w))]i|

i

= ϵ max |sgn([∇xf(x ; w)]i)|

i

ϵ

||δ2||∞ = max |[ϵ ∇xf(x ; w)]i|

i

= ϵ max |∇xf(x ; w)]i|.

i

The conditions under which ||δ1||∞ = ||δ2||∞ are when ∇xf(x ; w) has all its entries either positive or negative. In this case, the sgn() function doesn’t change the signs of the entries, and hence the magnitudes of the two vectors are equal. Otherwise, ||δ1||∞ is strictly less than ||δ2||∞.

The sgn() function gives us a guarantee on the ℓ∞ norm of the perturbation by ensuring that it is bounded by ϵ. In other words, ||δ1||∞ ≤ ϵ, which means that the perturbation applied to the input x is limited to a small range. This is important for ensuring that the adversarial example is visually similar to the original example and is imperceptible to human eyes.

1.1.2 Prediction under Attack [1pt]

If we remove the sgn() function from the FGSM, we are left with just the FGM

x′ ← x − ϵ ∇xf(x ; w)

Let us construct x′ using the FGM. Write down the model output under the adversarial attack f(x′ ; w) as a function of ϵ, x, w in a closed form.

ChatGPT: Using the FGM, we perturb x to obtain x′. The perturbation is given by ϵ∇x L(f(x ; w), y). In our case, since we are simply trying to reduce the output predicted by the classifier, we can set

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

y to be the opposite of the true label of x, i.e., if x is classified as positive, we set y = −1, and if x is classified as negative, we set y = 1. Thus, we have:

x′ = x + ϵ∇xL(f(x ; w), y)

x − ϵ∇xf(x ; w)y

x − ϵyw

where we used the fact that the gradient of a linear function f(x ; w) with respect to x is simply w.

Now, to obtain the predicted output of the classifier f(x′ ; w), we substitute the expression for x′:

f(x′ ; w) = w⊤x′

w⊤(x − ϵyw)

w⊤x − ϵyw⊤w

f(x ; w) − ϵy||w||2

Thus, we obtain the closed-form expression for the predicted output of the classifier under the adversarial attack x′ as:

f(x′ ; w) = f(x ; w) − ϵy||w||2

1.2 Gradient Descent and Weight Decay

The most trivial though impractical way of making a classifier robust to adversarial examples is to set w = 0 such that f(x ; w) = 0 for any x. However, this just computes a constant function, and is not useful. Intuitively, it looks like the smaller the norm of w, then less the output will change when changing x. We explore if this is always the case.

Suppose we have a design matrix X ∈ Rn×d where n is the number of samples and d is the dimensionality, and a target vector t ∈ Rn. We can define the objective of linear regression as

min

1

Xw

t

2

−

||2

w 2n||

If we penalize the squared ℓ2 norm of the weights, we end up with ridge regression:

w∗

= arg min

1

Xw

−

t 2

+ λ w

2,

2n

ridge

w

||

||2

||

||2

where λ is the weight decay coefficient,λ > 0.

1.2.1 Toy Example [0pt]

Consider the following dataset D = {((1, 1),

1

)((−1, −1), −1 )} visualized in Figure 1 for your

x1

t1

x2

t2

convenience. Draw the set of optimal solutions w∗ (it’s a line) in weight space (which is different from Figure 1 which is in data space) with w1∗ on the x-axis, and w2∗ on the y-axis. On the same plot, draw the contours of w⊤w. Is there a contour value for which the intersection between the set of optimal solutions and w⊤w is a single point? If so, provide the coorindate of the point. What does this imply about the uniqueness of ridge regression solution for this dataset?

5

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

Derive the analytical closed form of ϵ as a function of the weight decay coefficient λ such that

f(x′ ; wridge∗) = 0. Show your work. Does weight decay make the model more robust under FGM attack? Why?

(Hint: Substitute your 1.2.2 solution into 1.1.2 final form then set the equation to zero. Sim-plify.)

ChatGPT:

To derive the closed form of ϵ, we start by writing out the expression for the adversarial example using the update equation:

x′ = x − ϵ; ∇xf(x ; wridge∗)

x − ϵ; ∇x(xwridge) − ϵ; ∇xλwridge2

x − ϵwridge − 2ϵλw∗ridge

x − ϵ(1 + 2λ)wridge∗.

Now, we want to find the value of ϵ that satisfies f(x′ ; wridge∗) = 0. Using the expression for x′ and the definition of the linear classifier, we have:

f(x′ ; wridge∗) = (x − ϵ(1 + 2λ)wridge)wridge∗

xwridge − ϵ(1 + 2λ)wridge∗2.

Setting this expression to zero and solving for ϵ, we get:

xwridge = ϵ(1 + 2λ)w∗2ridge

xwridge

ϵ = (1 + 2λ)w∗2ridge.

This is the closed form expression for ϵ as a function of the weight decay coefficientλ that satisfies f(x′ ; wridge∗) = 0.

Now, let us consider the impact of weight decay on the adversarial robustness of the model. From the expression for ϵ, we see that the larger the value of λ, the smaller the value of ϵ for a

given input x and weight wridge∗. This means that weight decay is making the model more robust to adversarial attacks under FGM, as it requires a smaller perturbation to fool the model. Intuitively,

this is because weight decay encourages the model to have smaller weights, which in turn makes it less sensitive to small changes in the input. However, it is important to note that weight decay is not a silver bullet for adversarial robustness, and other techniques such as adversarial training may also be necessary to achieve robustness in practice.

1.2.4 The Adversary Strikes Back [0pt]

Now consider the 1-D case again under for the Fast Gradient Sign Method (FGSM) by including the sign function in the perturbation:

x′ ← x − ϵ sgn ∇xf(x ; wridge∗) .

Does weight decay make the model more robust under FGSM attack? Why?

Trading off Resources in Neural Net Training

2.1 Effect of batch size

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

2.1.1 Batch size vs. learning rate

Batch size affects the stochasticity in optimization, and therefore affects the choice of learning rate. We demonstrate this via a simple model called the noisy quadratic model (NQM). Despite the simplicity, the NQM captures many essential features in realistic neural network training Zhang et al. [2019].

For simplicity, we only consider the scalar version of the NQM. We have the quadratic loss L(w) = 12 aw2, where a > 0 and w ∈ R is the weight that we would like to optimize. Assume that we only have access to a noisy version of the gradient — each time when we make a query for the gradient, we obtain g(w), which is the true gradient L(w) with additive Guassian noise:

g(w) = L(w) + ϵ, ϵ ∼ N (0, σ2).

One way to reduce noise in the gradient is to use minibatch training. Let B be the batch size, and denote the minibatch gradient as gB(w):

1 B

i.i.d.

N(0, σ2).

gB(w) =

gi(w), where gi(w) = L(w) + ϵi, ϵi ∼

B

i=1

[1pt] As batch size increases, how do you expect the optimal learning rate to change? Briefly explain in 2-3 sentences.

(Hint: Think about how the minibatch gradient noise change with B.)

ChatGPT:

As the batch size increases, the variance of the minibatch gradient decreases, which implies that a larger learning rate can be used without the risk of overshooting the minimum. There-fore, as the batch size increases, the optimal learning rate tends to increase as well. However, the relationship between batch size and learning rate can also be affected by other factors, such as the curvature of the loss function and the specific optimization algorithm used.

2.1.2 Training steps vs. batch size

For most of neural network training in the real-world applications, we often observe the relationship of training steps and batch size for reaching a certain validation loss as illustrated in Figure 2.

[1pt] For the three points (A, B, C) on Figure 2, which one has the most efficient batch size (in terms of best resource and training time trade-off)? Assume that you have access to scalable (but not free) compute such that minibatches are parallelized efficiently. Briefly explain in 1-2 sentences.

[1pt] Figure 2 demonstrates that there are often two regimes in neural network training: the noise dominated regime and the curvature dominated regime. In the noise dominated regime, the bottleneck for optimization is that there exists a large amount of gradient noise. In the curvature dominated regime, the bottleneck of optimization is the ill-conditioned loss landscape. For points A and B on Figure 2, which regimes do they belong to, and what would you do to accelerate training? Fill each of the blanks with one best suited option.

Point A: Regime:

. Potential way to accelerate training:

.

Point B: Regime:

. Potential way to accelerate training:

.

Options:

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1


Figure 2: A cartoon illustration of the typical relationship between training steps and the batch size for reaching a certain validation loss (based on Shallue et al. [2018]). Learning rate and other related hyperparameters are tuned for each point on the curve.

Regimes: noise dominated / curvature dominated.

Potential ways to accelerate training: use higher order optimizers / seek parallel compute

2.2 Model size, dataset size and compute

We have seen in the previous section that batch size is an important hyperparameter during training. Besides efficiently minimizing the training loss, we are also interested in the test loss. Recently, researchers have observed an intriguing relationship between the test loss and hyperparameters such as the model size, dataset size and the amount of compute used. We explore this relationship for neural language models in this section. The figures in this question are from Kaplan et al. [2020].


Figure 3: Test loss of language models of different sizes, plotted against the dataset size (tokens processed) and the amount of compute (in petaflop/s-days).

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1


Figure 4: Test loss for different sized models after the initial transient period, plotted against the number of training steps (Smin) when using the critical batch sizes (the batch sizes that separate the two regimes in Question 2.1.2).

[1pt] Previously, you have trained a neural language model and obtained somewhat adequate per-formance. You have now secured more compute resources (in PF-days), and want to improve the model test performance (assume you will train from scratch). Which of the following is the best option? Give a brief explanation (2-3 sentences).

Train the same model with the same batch size for more steps.

Train the same model with a larger batch size (after tuning learning rate), for the same number of steps.

Increase the model size.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

Programming Assignment

What you have to submit for this part

For reference, here is everything you need to hand in:

This is the second half of your PDF report a3-writeup.pdf. Please include the solutions to the following problems. You may choose to export nmt.ipynb, bert.ipynb, clip.ipynb as a PDF and attach it to the first half of a3-writeup.pdf.

– Question 3: 3.1[Type 2] , 3.2[Type 2] , 3.3[Type 2]

– Question 4: 4.1[Type 1] , 4.3[Type 4] , 4.4[Type 4] .

– Question 5: 5.2[Type 4]

Your code filenmt.ipynb, bert.ipynb, clip.ipynb

Introduction

In this assignment, you will explore common tasks and model architectures in Natural Language Processing (NLP). Along the way, you will gain experience with important concepts like attention mechanisms (Section 3), pretrained language models (Section 4) and multimodal vision and language models (Section 5).

Setting Up

We recommend that you use Colab(https://colab.research.google.com/) for the assignment. To setup the Colab environment, just open the notebooks for each part of the assignment and make a copy in your own Google Drive account.

Deliverables

Each section is followed by a checklist of deliverables to add in the assignment writeup. To also give a better sense of our expectations for the answers to the conceptual questions, we’ve put maximum sentence limits. You will not be graded for any additional sentences.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

Neural machine translation (NMT) [2pt]

Neural machine translation (NMT) is a subfield of NLP that aims to translate between languages using neural networks. In this section, will we train a NMT model on the toy task of English → Pig Latin. Please read the following background section carefully before attempting the questions.

Background

The task

Pig Latin is a simple transformation of English based on the following rules:

If the first letter of a word is a consonant, then the letter is moved to the end of the word, and the letters “ay” are added to the end: team → eamtay.

If the first letter is a vowel, then the word is left unchanged and the letters “way” are added to the end: impress → impressway.

In addition, some consonant pairs, such as “sh”, are treated as a block and are moved to the end of the string together: shopping → oppingshay.

To translate a sentence from English to Pig-Latin, we apply these rules to each word independently:

i went shopping → iway entway oppingshay

Our goal is to build a NMT model that can learn the rules of Pig-Latin implicitly from (English, Pig-Latin) word pairs. Since the translation to Pig Latin involves moving characters around in a string, we will use character-level transformer model. Because English and Pig-Latin are similar in structure, the translation task is almost a copy task; the model must remember each character in the input and recall the characters in a specific order to produce the output. This makes it an ideal task for understanding the capacity of NMT models.

The data

The data for this task consists of pairs of words {(s(i), t(i))}Ni=1 where the source s(i) is an English word, and the target t(i) is its translation in Pig-Latin.4 The dataset contains 3198 unique (English, Pig-Latin) pairs in total; the first few examples are:

{ (the, ethay), (family, amilyfay), (of, ofway), … }

In this assignment, you will investigate the effect of dataset size on generalization ability. We provide a small and large dataset. The small dataset is composed of a subset of the unique words from the book “Sense and Sensibility” by Jane Austen. The vocabulary consists of 29 tokens: the 26 standard alphabet letters (all lowercase), the dash symbol -, and two special tokens <SOS> and <EOS> that denote the start and end of a sequence, respectively.5 The second, larger dataset is obtained from Peter Norvig’s natural language corpus.6 It contains the top 20,000 most used English words, which is combined with the previous data set to obtain 22,402 unique words. This dataset contains the same vocabulary as the previous dataset.

In order to simplify the processing of mini-batches of words, the word pairs are grouped based on the lengths of the source and target. Thus, in each mini-batch, the source words are all the same length, and the target words are all the same length. This simplifies the code, as we don’t have to worry about batches of variable-length sequences.

5Note that for the English-to-Pig-Latin task, the input and output sequences share the same vocabulary; this is not always the case for other translation tasks (i.e., between languages that use different alphabets)

6https://norvig.com/ngrams/

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

The model


Figure 5: The transformer architecture. Vaswani et al. [2017]

Translation is a sequence-to-sequence (seq2seq) problem. The goal is to train a model to transform one sequence into another. A transformer model Vaswani et al. [2017] uses an encoder-decoder architecture and relies entirely on an attention mechanism to draw global dependencies between the input sequence and the output sequence. The encoder processes the input sequence in parallel using stacked self-attention and point-wise fully connected layers, as shown in Figure 5. Given the hidden representations of each input token processed through an encoder, the decoder then generates an output sequence one at a time. The model is auto-regressive when generating the output tokens.

Specifically, input characters are passed through an embedding layer before being fed into an encoder model. If H is the dimension of the encoder hidden state, we learn a 29 × H embedding matrix, where each of the 29 characters in the vocabulary is assigned a H-dimensional embedding. At each time step, the decoder outputs a vector of unnormalized log probabilities given by a linear transformation of the decoder hidden state. When these probabilities are normalized (i.e. by passing them through a softmax), they define a distribution over the vocabulary, indicating the most probable characters for that time step. The model is trained via a cross-entropy loss between

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

the decoder distribution and ground-truth at each time step.

3.1 Transformers for NMT (Attention Is All You Need) [4pt]

In order to answer the following questions correctly, please make sure that you have run the code from nmt.ipynb, Part1, Training and evaluation code prior to answering the following questions.

[0.5pt] In lecture, we learnt about Scaled Dot-product Attention used in the transformer models. The function f is a dot product between the linearly transformed query and keys using weight matrices Wq and Wk:

(t)

(WqQt)T

(WkKi)

α˜i

= f(Qt, Ki) =

√

,

d

αi(t) = softmax(˜α(t))i,

T

ct

= α(t)WvVi,

i

i=1

where, d is the dimension of the query and the Wv denotes weight matrix project the value to produce the final context vectors.

Implement the scaled dot-product attention mechanism. Fill in the forward meth-ods of the ScaledDotAttention class. Use the PyTorch torch.bmm (or @) to compute the dot product between the batched queries and the batched keys in the forward pass of the ScaledDotAttention class for the unnormalized attention weights.

The following functions are useful in implementing models like this. You might find it useful to get familiar with how they work. (click to jump to the PyTorch documentation):

squeeze

unsqueeze

expand as

cat

view

bmm (or @)

Your forward pass needs to work with both 2D query tensor (batch_size x (1) x hidden_size) and 3D query tensor (batch_size x k x hidden_size).

[0.5pt] Implement the causal scaled dot-product attention mechanism. Fill in the forward method in the CausalScaledDotAttention class. It will be mostly the same as the ScaledDotAttention class. The additional computation is to mask out the attention to the future time steps. You will need to add self.neg_inf to some of the entries in the unnormalized attention weights. You may find torch.tril or torch.triu handy for this part.

[0.5pt] We will now use ScaledDotAttention as the building blocks for a simplified trans-former Vaswani et al. [2017] encoder.

The encoder looks like the left half of Figure 5. The encoder consists of three components:

14

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

Positional encoding: To encode the position of each word, we add to its embedding a constant vector that depends on its position:

pth word embedding = input embedding + positional encoding(p)

We follow the same positional encoding methodology described in Vaswani et al. [2017].

That is we use sine and cosine functions:

pos

PE(pos, 2i) = sin

(3.1)

100002i/dmodel

pos

PE(pos, 2i + 1) = cos

(3.2)

100002i/dmodel

Since we always use the same positional encodings throughout the training, we pre-generate all those we’ll need while constructing this class (before training) and keep reusing them throughout the training.

A ScaledDotAttention operation.

A following MLP.

For this question, describe why we need to represent the position of each word through this positional encoding in one or two sentences. Additionally, describe the advantages of using this positional encoding method, as opposed to other positional encoding methods such as a one hot encoding in one or two sentences.

[1pt] In the code notebook, we have provided an experimental setup to evaluate the perfor-mance of the Transformer as a function of hidden size and data set size. Run the Transformer model using hidden size 32 versus 64, and using the small versus large dataset (in total, 4 runs). We suggest using the provided hyper-parameters for this experiment.

Run these experiments, and report the effects of increasing model capacity via the hidden size, and the effects of increasing dataset size. In particular, report your observations on how loss as a function of gradient descent iterations is affected, and how changing model/dataset size affects the generalization of the model. Are these results what you would expect?

In your report, include the two loss curves output by save_loss_comparison_by_hidden and save_loss_comparison_by_dataset, the lowest attained validation loss for each run, and your response to the above questions.

Deliverables

Create a section in your report called Scaled Dot Product Attention. Add the following:

Screenshots of your ScaledDotProduct, CausalScaledDotProduct implementations. High-light the lines you’ve added. [1pt]

Your answer to question 3. [0.5pt]

The two loss curves plots output by the experimental setup in question 4, and the lowest validation loss for each run. [1pt]

Your response to the written component of question 4. Your analysis should not exceed six sentences. [1pt]

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

3.2 Decoder Only NMT

In this subsection, we will train a decoder-only NMT model using the CausalAttention mechanism. The key difference between this approach and the previous encoder-decoder approach is that we do not encode a hidden state of the input sequence first using an encoder. Instead, we feed both the input sequence and the target sequence to a decoder simultaneously, as in Figure 6. The input sequence and the target sequence will be separated using an end-of-prompt token (EOP). The concatenated input to the decoder will have SOS token added at the beginning, and the concatenated target will have EOS token added at the end. In our provided notebook, the decoder will process this concatenated input using causal attention, but we compute the cross-entropy loss by using the output tokens from the output of <EOP> only.


Figure 6: Training the decoder-only NMT model.

For test-time translations, we first feed the input sequence to a trained decoder, enclosed by a SOS token and a EOP token, as shown in Figure 7. We obtain the first translated token a in this case and concatenate the input sequence with the generated token. Then we feed the concatenated sequence to the decoder and obtain two tokens a and t. This procedure is repeated until reaching the maximum target length or generating a <EOS> token.


Figure 7: Translating a text using the decoder-only NMT model.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

In order to answer the following questions correctly, please make sure that you have run the code from nmt.ipynb, Part2, Training and evaluation code prior to answering the fol-lowing questions.

[1pt] Construct the input tensors and the target tensors for training a decoder. For this ques-tion, we ask you to implement the function generate_tensors_for_training_decoder_nmt that takes in an input sequence plus an end-of-prompt token and an output sequence plus an end-of-sentence token and returns two concatenated sequences. One has the form

<SOS> input sequence <EOP> output sequence

as in the input to the decoder shown in Figure 6, and the other has the form input sequence <EOP> output sequence <EOS>

[1pt] Implement the forward function in DecoderOnlyTransformer.

[1pt] Train the model. Now, run the training and testing code block to see the generated translation using a decoder-only model. Comment on the pros and cons of the decoder-only approach. How is the quality of your generated results compared to the ones using the encoder-decoder model?

Deliverables

Create a section in your report called Decoder Only NMT. Add the following:

Your answer to question 1. (Screenshots of your implementations) [1.0pt]

Your answer to question 2. (Screenshots of your implementations) [1.0pt]

Your written response to the question 3. [1.0pt]

3.3 Scaling Law and IsoFLOP Profiles

This section will give you hands-on experience charting scaling law curves to forecast neural network performance. Scaling law is a fundamental concept that describes how the performance of a neural network changes with its size. Specifically, it relates the number of parameters or computations required by a neural network to achieve a certain level of performance, such as accuracy or loss. The scaling law provides a useful tool for predicting the performance of neural networks as they are scaled up or down.

IsoFLOP is a method proposed in the ”Training Compute-Optimal Large Language Models” paper [Hoffmann et al., 2022] to study the scaling law of large language models. The authors of the paper used IsoFLOP to study the effect of model size on the performance of large language models and to determine the optimal model size that maximizes performance for a given computational budget.

The motivation for using IsoFLOP to forecast neural network performance is twofold. Firstly, it provides a more accurate and efficient way to explore the scaling law of large language models than traditional methods, which involve training multiple models at different sizes. Secondly, IsoFLOP allows for a better understanding of the trade-off between model size and training cost, which is crucial for designing large-scale neural network architectures that are both efficient and effective. By

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

leveraging IsoFLOP, researchers can gain insights into the scaling properties of neural networks, such as their accuracy and computational efficiency, and optimize their performance for specific applications and computational resources.

In this question, we will plot the scaling law curve for the decoder-only translation models from the previous section. The notebook provided trains six translation models with different model sizes and varies the FLOP counts by training for different numbers of epochs. You are asked to complete the functions to make the final IsoFLOP curve consisting of models ranging from 0.08 TFLOPs to 1.28 TFLOPs.

[0.5pt] Train six decoder-only translation models using the code provided and plot the vali-dation loss as the function of FLOPs. Comment on any interesting thing you observe. Does larger model always have a smaller validation loss? (Hint: See Question 2.2)

[1pt] IsoFLOP Profiles. For a given FLOPs, fit a quadratic function to the validation loss and number of parameters in the log space. Find the optimal number of parameters using the quadratic function. Specifically, you need to fill the “findoptimal params” function.

[1pt] Complete the Compute Optimal Model plot by fitting a linear line to the target FLOPs and the optimal model parameters. Based on the plot, estimate the optimal number of parameters when we have a compute budget of 1e15.

[1pt] Plot Compute Optimal Token using the code provided. Now, given the Compute Optimal Model plot and Compute Optimal Token plot, is the training setup in Section 3.2.3 compute optimal? If not, how should we change it?

Deliverables

Create a section in your report called Scaling Law and IsoFLOP Profiles. Add the following:

Your written response to the question 1. Your answer should not exceed 3 sentences. [0.5pt]

Your answer to question 2. (Screenshots of your implementations) [1.0pt]

Your answer to question 3. (Screenshots of your implementations). The optimal number of parameters given 1e15 FLOPs and the process of how you estimate it. [1.0pt]

Your written response to the question 4. Your answer should not exceed 3 sentences. [1.0pt]

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

Fine-tuning Pretrained Language Models (LMs) [2pt]

The previous sections had you train models from scratch. However, similar to computer vision (CV), it is now very common in natural language processing (NLP) to fine-tune pretrained models. Indeed, this has been described as “NLP’s ImageNet moment.”7 In this section, we will learn how to fine-tune pretrained language models (LMs) on a new task. We will use a simple classification task, where the goal is to determine whether a verbal numerical expression is negative (label 0), zero (label 1), or positive (label 2). For example, “eight minus ten” is negative, so our classifier should output label index 0. As our pretrained LM, we will use the popular BERT model, which uses a transformer encoder architecture. More specifically, we will explore two versions of BERT: MathBERT [Shen et al., 2021], which has been pretrained on a large mathematical corpus ranging from pre-kindergarten to college graduate level mathematical content and BERTweet [Nguyen et al., 2020], which has been pretrained on 100s of millions of tweets.

Most of the code is given to you in the notebook https://colab.research.google.com/ github/uoft-csc413/2022/blob/master/assets/assignments/bert.ipynb. The starter code uses the HuggingFace Transformers library8, which has more than 50k stars on GitHub due to its ease of use, and will be very useful for your NLP research or projects in the future. Your task is to adapt BERT so that it can be fine-tuned on our downstream task. Before starting this section, please carefully review the background for BERT and the verbal arithmetic dataset (below).

Background

BERT

Bidirectional Encoder Representations from Transformers (BERT) [Devlin et al., 2019] is a LM based on the Transformer [Vaswani et al., 2017] encoder architecture that has been pretrained on a large dataset of unlabeled sentences from Wikipedia and BookCorpus [Zhu et al., 2015]. Given a sequence of tokens, BERT outputs a “contextualized representation” vector for each token. Because BERT is pretrained on a large amount of text, these contextualized representations encode useful properties of the syntax and semantics of language.

BERT has 2 pretraining objectives: (1) Masked Language Modeling (MLM), and (2) Next Sentence Prediction (NSP). The input to the model is a sequence of tokens of the form:

[CLS] Sentence A [SEP] Sentence B

where [CLS] (“class”) and [SEP] (“separator”) are special tokens. In MLM, some percentage of the input tokens are randomly “masked” by replacing them with the [MASK] token, and the objective is to use the final layer representation for that masked token to predict the correct word that was masked out9. In NSP, the task is to use the contextualized representation of the [CLS] token to predict whether sentence A and sentence B are consecutive sentences in the unlabeled dataset. See Figure 8 for the conceptual picture of BERT pretraining and fine-tuning.

Once pretrained, we can fine-tune BERT on a downstream task of interest, such as sentiment analysis or question-answering, benefiting from its learned contextual representations. Typically, this is done by adding a simple classifier, which maps BERTs outputs to the class labels for our downstream task. Often, this classifier is a single linear layer + softmax. We can choose to train

https://ruder.io/nlp-imagenet/

https://huggingface.co/docs/transformers

9The actual training setup is slightly more complicated but conceptually similar. Notice, this is similar to one of the models in Programming Assignment 1!

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1


CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

Verbal Arithmetic Dataset

The verbal arithmetic dataset contains pairs of input sentences and labels. The input sentences express a simple addition or subtraction. Each input is labelled as 0, 1, or 2 if it evaluates to negative, zero, or positive, respectively. There are 640 examples in the train set and 160 in the test set. All inputs have only three tokens similar to the examples shown below:

Input expression

Label

Label meaning

four minus ten

0

“negative”

eighteen minus eighteen

1

“zero”

four plus seven

2

“positive”

Questions:

[1pt] Add a classifier to BERT. Open the notebook https://colab.research.google.com/ github/uoft-csc413/2022/blob/master/assets/assignments/bert.ipynb and complete Question 1 by filling in the missing lines of code inBertForSentenceClassification.

[0pt] Fine-tune BERT. Open the notebook and run the cells under Question 2 to fine-tune the BERT model on the verbal arithmetic dataset. If question 1 was completed correctly, the model should train, and a plot of train loss and validation accuracy will be displayed.

[0.5pt] Freezing the pretrained weights. Open the notebook and run the cells under Question 3 to fine-tune only the classifiers weights, leaving BERTs weights frozen. After training, answer the following questions (no more than four sentences total)

Compared to fine-tuning (see Question 2), what is the effect on train time when BERTs weights are frozen? Why? (1-2 sentences)

Compared to fine-tuning (see Question 2), what is the effect on performance (i.e. vali-dation accuracy) when BERTs weights are frozen? Why? (1-2 sentences)

[0.5pt] Effect of pretraining data. Open the notebook and run the cells under Question 4 in order to repeat the fine-tuning process using the pretrained weights of BERTweet. After training, answer the following questions (no more than three sentences total).

Compared to fine-tuning BERT with the pretrained weights from MathBERT (see Ques-tion 2), what is the effect on performance (i.e. validation accuracy) when we fine-tune BERT with the pretrained weights from BERTweet? Why might this be the case? (2-3 sentences)

[0pt] Inspect models predictions. Open the notebook and run the cells under Question 5. We have provided a function that allows you to inspect a models predictions for a given input. Can you find examples where one model clearly outperforms the others? Can you find examples where all models perform poorly?

Deliverables:

The completed BertForSentenceClassification. Either the code or a screenshot of the

code. Make sure both the init and forward methods are clearly visible. [1pt]

Answer to question 3. Your answer should not exceed 4 sentences. [0.5pt]

Answer to question 4. Your answer should not exceed 3 sentences. [0.5pt]

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

Connecting Text and Images with CLIP [1pt]

Throughout this course, we have seen powerful image models and expressive language models. In this section, we will connect the two modalities by exploring CLIP, a model trained to predict an image’s caption to learn better image representations.


Figure 10: 1. Contrastive pre-training task that predicts the caption that corresponds to an image out of many possible captions. 2. At test time, each class is converted to a caption. This is used with 3. as a zero-shot classifier for a new image that predicts the best (image, caption) pair. Figure taken from [Radford et al., 2021a]

Background for CLIP:

The motivation behind Contrastive Language-Image Pre-training (CLIP) [Radford et al., 2021b] was to leverage information from natural language to improve zero-shot classification of images. The model is pre-trained on 400 million (image, caption) pairs collected from the internet on the following task: given the image, predict which caption was paired with it out of 32,768 randomly sampled captions (Figure 10). This is done by first computing the feature embedding of the image and feature embeddings of possible captions. The cosine similarity of the embeddings is computed and converted into a probability distribution. The outcome is that the network learns many visual concepts and associates them with a name.

At test time, the model is turned into a zero-shot classifier: all possible classes are converted to a caption such as ”a photo of a (class)” and CLIP estimates the best (image, caption) pair for a new image. Overall, CLIP offers many significant advantages: it does not require expensive hand-labelling while achieving competitive results and offers greater flexibility and generalizability over existing ImageNet models.

Questions:

[0pt] Interacting with CLIP. Open the notebook https://colab.research.google.com/ github/uoft-csc413/2022/blob/master/assets/assignments/clip.ipynb. Read through Section I and run the code cells to get familiar with CLIP.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

[1pt] Prompting CLIP. Complete Section II. Come up with a caption that will “prompt” CLIP to select the following target image:


Figure 11: Image that should be selected by CLIP.

Comment on the process of finding the caption: was it easy, or were there any difficulties? (no more than one sentence)

Deliverables:

The caption you wrote that causes CLIP to select the image in Figure 11, as well as a brief (1 sentence) comment on the search process. [1pt]

What you need to submit

The completed notebook files: nmt.ipynb, bert.ipynb, clip.ipynb.

A PDF document titled a3-writeup.pdf containing your answers to the conceptual questions. You may directly append the PDF exports of the notebooks into the final a3-writeup.pdf.

References

Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

Guodong Zhang, Lala Li, Zachary Nado, James Martens, Sushant Sachdeva, George E Dahl, Christopher J Shallue, and Roger Grosse. Which algorithmic choices matter at which batch sizes? insights from a noisy quadratic model. arXiv preprint arXiv:1907.04164, 2019.

Christopher J Shallue, Jaehoon Lee, Joseph Antognini, Jascha Sohl-Dickstein, Roy Frostig, and George E Dahl. Measuring the effects of data parallelism on neural network training. arXiv preprint arXiv:1811.03600, 2018.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Jia Tracy Shen, Michiharu Yamashita, Ethan Prihar, Neil Heffernan, Xintao Wu, Ben Graff, and Dongwon Lee. Mathbert: A pre-trained language model for general nlp tasks in mathematics education. arXiv preprint arXiv:2106.07340, 2021.

Dat Quoc Nguyen, Thanh Vu, and Anh Tuan Nguyen. Bertweet: A pre-trained language model for english tweets. arXiv preprint arXiv:2005.10200, 2020.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https: //www.aclweb.org/anthology/N19-1423.

Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pages 19–27, 2015.

Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, and Sandhini Agarwal. Clip:

Connecting text and images, Jan 2021a. URL https://openai.com/blog/clip/.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agar-wal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021b.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1


CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1


CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1


Figure 14: Question 1.2.2

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1


CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1


Figure 16: Question 2.1.1

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1


Figure 17: Question 3.1

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 1


Figure 18: Question 3.2
