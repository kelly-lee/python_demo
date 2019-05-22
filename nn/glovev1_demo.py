#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import sys

# !/usr/bin/env python

from argparse import ArgumentParser
import codecs
from collections import Counter
import itertools
from functools import partial
import logging
from math import log
import os.path
import cPickle as pickle
from random import shuffle

import msgpack
import numpy as np
from scipy import sparse

from util import listify

logger = logging.getLogger("glove")


def build_vocab(corpus):
    """
    Build a vocabulary with word frequencies for an entire corpus.
    Returns a dictionary `w -> (i, f)`, mapping word strings to pairs of
    word ID and word corpus frequency.
    """

    logger.info("Building vocab from corpus")

    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)

    logger.info("Done building vocab from corpus.")

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.iteritems())}


@listify
def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    """
    Build a word co-occurrence list for the given corpus.
    This function is a tuple generator, where each element (representing
    a cooccurrence pair) is of the form
        (i_main, i_context, cooccurrence)
    where `i_main` is the ID of the main word in the cooccurrence and
    `i_context` is the ID of the context word, and `cooccurrence` is the
    `X_{ij}` cooccurrence value as described in Pennington et al.
    (2014).
    If `min_count` is not `None`, cooccurrence pairs where either word
    occurs in the corpus fewer than `min_count` times are ignored.
    """

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.iteritems())

    # Collect cooccurrences internally as a sparse matrix for passable
    # indexing speed; we'll convert into a list later
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            logger.info("Building cooccurrence matrix: on line %i", i)

        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

        for center_i, center_id in enumerate(token_ids):
            # Collect all word IDs in left window of center word
            context_ids = token_ids[max(0, center_i - window_size): center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    # Now yield our tuple sequence (dig into the LiL-matrix internals to
    # quickly iterate through all nonzero cells)
    for i, (row, data) in enumerate(itertools.izip(cooccurrences.rows,
                                                   cooccurrences.data)):
        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue

            yield i, j, data[data_idx]


def run_iter(vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.
    `data` is a pre-fetched data / weights list where each element is of
    the form
        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)
    as produced by the `train_glove` function. Each element in this
    tuple is an `ndarray` view into the data structure which contains
    it.
    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.
    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.
    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """

    global_cost = 0

    # We want to iterate over data randomly so as not to unintentionally
    # bias the word vector contents
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:
        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context)
                      + b_main[0] + b_context[0]
                      - log(cooccurrence))

        # Compute cost
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost

        # Compute gradients for word vector terms.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word, biases, etc.
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        # Compute gradients for bias terms
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
            gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cost


def train_glove(vocab, cooccurrences, iter_callback=None, vector_size=100,
                iterations=25, **kwargs):
    """
    Train GloVe vectors on the given generator `cooccurrences`, where
    each element is of the form
        (word_i_id, word_j_id, x_ij)
    where `x_ij` is a cooccurrence value $X_{ij}$ as presented in the
    matrix defined by `build_cooccur` and the Pennington et al. (2014)
    paper itself.
    If `iter_callback` is not `None`, the provided function will be
    called after each iteration with the learned `W` matrix so far.
    Keyword arguments are passed on to the iteration step function
    `run_iter`.
    Returns the computed word vector matrix `W`.
    """

    vocab_size = len(vocab)

    # Word vector matrix. This matrix is (2V) * d, where N is the size
    # of the corpus vocabulary and d is the dimensionality of the word
    # vectors. All elements are initialized randomly in the range (-0.5,
    # 0.5]. We build two word vectors for each word: one for the word as
    # the main (center) word and one for the word as a context word.
    #
    # It is up to the client to decide what to do with the resulting two
    # vectors. Pennington et al. (2014) suggest adding or averaging the
    # two for each word, or discarding the context vectors.
    W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

    # Bias terms, each associated with a single vector. An array of size
    # $2V$, initialized randomly in the range (-0.5, 0.5].
    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # Training is done via adaptive gradient descent (AdaGrad). To make
    # this work we need to store the sum of squares of all previous
    # gradients.
    #
    # Like `W`, this matrix is (2V) * d.
    #
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    # Build a reusable list from the given cooccurrence generator,
    # pre-fetching all necessary data.
    #
    # NB: These are all views into the actual data matrices, so updates
    # to them will pass on to the real data structures
    #
    # (We even extract the single-element biases as slices so that we
    # can use them as views)
    data = [(W[i_main], W[i_context + vocab_size],
             biases[i_main: i_main + 1],
             biases[i_context + vocab_size: i_context + vocab_size + 1],
             gradient_squared[i_main],
             gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main: i_main + 1],
             gradient_squared_biases[i_context + vocab_size
                                     : i_context + vocab_size + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]

    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)

        cost = run_iter(vocab, data, **kwargs)

        logger.info("\t\tDone (cost %f)", cost)

        if iter_callback is not None:
            iter_callback(W)

    return W


def save_model(W, path):
    with open(path, 'wb') as vector_f:
        pickle.dump(W, vector_f, protocol=2)

    logger.info("Saved vectors to %s", path)


def test():
    # Mock corpus (shamelessly stolen from Gensim word2vec tests)
    test_corpus = ("""human interface computer
    survey user computer system response time
    eps user interface system
    system human system eps
    user response time
    trees
    graph trees
    graph minors trees
    graph minors survey
    I like graph and stuff
    I like trees and stuff
    Sometimes I build a graph
    Sometimes I build trees""").split("\n")

    vocab = build_vocab(test_corpus)
    cooccur = build_cooccur(vocab, test_corpus, window_size=10)
    # id2word = evaluate.make_id2word(vocab)

    W = train_glove(vocab, cooccur, vector_size=10, iterations=500)

    # Merge and normalize word vectors
    # W = evaluate.merge_main_context(W)

    # save_model(W, arguments.vector_path)


if __name__ == '__main__':
    test()
