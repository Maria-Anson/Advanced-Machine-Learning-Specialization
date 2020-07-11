## Image Captioning Final Project

*Warning! This assignment requires several hours of computation, because it
contains training of a real world neural network architecture on a CPU, which we
think is essential for learning deep learning. We recommend to use offline
instructions at the bottom to setup your own environment (Docker or Anaconda).
If it's not an option, you can proceed with Coursera Notebooks, but do expect
kernel interruptions. In the meantime we're adding model checkpointing to our
tasks, this should help to continue from where you left off. We think the time
to setup your own environment is worth the comfort throughout the course.*

In this final project you will define and train an image-to-caption model, that
can produce descriptions for real world images!

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/I_VlgpvaEeeHrwpWBTEPxg_e316c4ae2d37150269408441640fb441_encoder_decoder.png?expiry=1594598400000&hmac=3fYUUD4L50ij3MhM9zlop0zJLTm0mPXcRp0Vkkbj4uQ)

Model architecture: CNN encoder and RNN decoder.
([https://research.googleblog.com/2014/11/a-picture-is-worth-thousand-coherent.html](https://research.googleblog.com/2014/11/a-picture-is-worth-thousand-coherent.html))

You will see  in some cells that are meant for grading. You must run them to
submit your answers to graded assignment parts. You can send partial answers to
see your progress on the task.

You will be asked to implement decoder architecture and train the model. 

We will also ask you to pick 10 examples where the model gives good results and
10 examples where it fails (you can use validation sample or download images
from Internet).

**Your solution will be graded in two ways:**

1. We run automatic tests for your implementation using graded assignment.

2. We ask your peers to review the caption examples that you've provided, just
to check that the model really works.

Good luck!

Coursera Jupyter Environment can be slow if many learners use it heavily. Our
tasks are compute-heavy and we recommend to run them on your hardware for
optimal performance. For instructions follow
[https://github.com/hse-aml/intro-to-dl/blob/master/README.md](https://github.com/hse-aml/intro-to-dl/blob/master/README.md)

*****
