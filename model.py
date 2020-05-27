# inbuilt lib imports:
from typing import Dict
import math
import numpy as np
# external libs
import tensorflow as tf
from tensorflow.keras import models, layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        
        return tf.pow(vector,3)
        
        #raise NotImplementedError
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_transitions = num_transitions
        self.regularization_lambda = regularization_lambda 
        self.trainable_embeddings = trainable_embeddings
        
        #comment the below line in order to train with fixed embeddings or without trainable embeddings
        self.embeddings = tf.Variable(tf.zeros([self.vocab_size, self.embedding_dim]))
        
        #uncomment the below line in order to train with fixed embeddings or without trainable embeddings
        #self.embeddings = tf.Variable(tf.zeros([self.vocab_size, self.embedding_dim]),trainable=False)

        self.w_inp = tf.Variable(tf.random.truncated_normal([self.num_tokens*self.embedding_dim, self.hidden_dim],stddev=1.0 / math.sqrt(self.hidden_dim)))
        
        self.w_out = tf.Variable(tf.random.truncated_normal([self.hidden_dim, self.num_transitions], stddev=1.0 / math.sqrt(self.num_transitions)))
        
        self.bias_inp = tf.Variable(tf.zeros([self.hidden_dim]))
        
        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
           
        # TODO(Students) Start
        embedings_trained = tf.nn.embedding_lookup(self.embeddings, inputs)
        
        embedings_trained = tf.reshape(embedings_trained,[inputs.shape[0], self.num_tokens*self.embedding_dim])
        
        val_1 = tf.add(tf.matmul(embedings_trained, self.w_inp),self.bias_inp)

        activated_val = self._activation(val_1)
                
        res = tf.matmul(activated_val, self.w_out)
        
        logits = res
        
        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        
        regularization = self.regularization_lambda
        softmax = tf.nn.softmax(logits) #softmax logits
        train_labels = tf.nn.relu(labels) #masking        
                
        regularization = 0.5 * self.regularization_lambda * (tf.nn.l2_loss(self.w_inp) + tf.nn.l2_loss(self.w_out) + tf.nn.l2_loss(self.bias_inp) + tf.nn.l2_loss(self.embeddings)) #regularization
       
        cce = tf.keras.losses.CategoricalCrossentropy() 
        loss = cce(train_labels,softmax) #cross entropy loss function
        
        # TODO(Students) End

        return loss + regularization
