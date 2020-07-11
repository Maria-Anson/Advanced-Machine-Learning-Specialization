# How to use RNNs
> 
> Latest Submission Grade
> 
> 100%
> 
>  1.Question 1
> 
> Consider RNNs for five different types of tasks:
> 
> 1.  Element-wise sequence classification
> 2.  Unconditional sequence generation
> 3.  Sequence classification
> 4.  Conditional sequence generation
> 5.  Sequence translation
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/j6SSNp5XEee8kwqP-vsHDg_173b2f330915b24cf0a108a955db0b34_Screen-Shot-2017-09-21-at-01.52.59.png?expiry=1594598400000&hmac=hzhepgI1lsKiePO5o_CiSfAlGU8yAaWoPTQUR1ejbGk)
> 
> Which of these RNNs is the most suitable one to solve the task of music generation from scratch?
> 
> 1 / 1 point 
> 
>  1\. Element-wise sequence classification 
> 

      2\. Unconditional sequence generation 
> 
>  3\. Sequence classification 
> 
>  4\. Conditional sequence generation 
> 
>  5\. Sequence translation 
> 
> Check
> 
> Correct
> 
> That is it!
> 
>  2.Question 2
> 
> Consider 5 different RNNs from the previous question. Which of these RNNs is the most suitable one to solve the task of music generation from notes?
> 
> 1 / 1 point 
> 
>  1\. Element-wise sequence classification 
> 
>  2\. Unconditional sequence generation 
> 
>  3\. Sequence classification 
> 
>  4\. Conditional sequence generation 
> 

      5\. Sequence translation 
> 
> Check
> 
> Correct
> 
> Yes, this is a translation from notes to audio.
> 
>  3.Question 3
> 
> Consider 5 different RNNs from the first question. We want to generate music from scratch and additionally, each generated sample should be from a specific instrument. Which of these RNNs is the most suitable one to solve this task?
> 
> 1 / 1 point 
> 
>  1\. Element-wise sequence classification 
> 
>  2\. Unconditional sequence generation 
> 
>  3\. Sequence classification 
> 

      4\. Conditional sequence generation 
> 
>  5\. Sequence translation 
> 
> Check
> 
> Correct
> 
> That is it!
> 
>  4.Question 4
> 
> Choose correct statements about image captioning architecture from the lecture:
> 
> ![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/3Z1XfJ5eEeeENhLqXb0-qg_b639a15a09ee8782ebc4c835fa3cbf88_Screen-Shot-2017-09-21-at-02.47.44.png?expiry=1594598400000&hmac=O_h3YQXxtaHIDtG0X18mv6p5za25mYBc7X6tCY6cV0A)
> 
> 1 / 1 point 
> 

      Any CNN may be used to represent an image with a feature vector. 
> 
> Check
> 
> Correct
> 
> We can use any CNN but the stronger, the better!
> 
>  This is a sequence-to-sequence architecture (sequence translation (5) from the first question). 
> 
>  There is no benefit in pre-training of any part of this model. 
> 

      It is possible to train this model end-to-end without pretraining. 
> 
> Check
> 
> Correct
> 
> It is possible if we have a big enough dataset of images with captions. But in practice, the CNN part is usually pretrained on a big dataset of images without captions. The reasons are:
> 
> 1.  Datasets of images without captions are much bigger and we need a lot of images to train a sophisticated CNN.
> 2.  Separate training of the CNN is much faster.
> 
>  5.Question 5
> 
> Suppose Nick has a trained sequence-to-sequence machine translation model. He wants to generate the translation for a new sentence in the way that this translation has the highest probability in the model. To do this at each time step of the decoder he chooses the most probable next word instead of the generating from the distribution. Does this scheme guaranty that the resulting output sentence is the most probable one?
> 
> 1 / 1 point 
> 
>  Yes 
> 

      No 
> 
> Check
> 
> Correct
> 
> Unfortunately, it is true. That is why beam search is usually used to generate more probable sequences.
>
> -- https://www.coursera.org/learn/intro-to-deep-learning/exam/3s1iS/how-to-use-rnns/attempt?redirectToCover=true#Tunnel Vision Close
