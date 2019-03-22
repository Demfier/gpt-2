import os
import json
import math
import model
import pickle
import encoder
import numpy as np
from tqdm import tqdm
import tensorflow as tf


def test_sequence(*, hparams, X, length):
    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens,
                                past=past, reuse=tf.AUTO_REUSE)
        logits = lm_output['logits'][:, :, :hparams.n_vocab][0]
        logits = tf.nn.softmax(logits)
        return {
            'logits': logits,
        }

    with tf.name_scope('test_sequence'):
        context_output = step(hparams, X)
        return context_output['logits']


def load_sentences(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()


def evaluate_model(model_name='117M', seed=None, batch_size=1):

    enc = encoder.get_encoder(model_name)
    eos_token = enc.encoder['<|endoftext|>']
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    with tf.Session(graph=tf.Graph()) as sess:
        tokens = tf.placeholder(tf.int32, [batch_size, None])

        probs = test_sequence(
            hparams=hparams, X=tokens, length=tokens.shape[1])

        # Load pretrained model
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        data_path = '../../generated_files/current_evaluation/'
        corpus_ppls = {}
        files = os.listdir(data_path)
        for file in files:
            # sentence level PPLs
            ppl_1 = []
            ppl_2 = []
            ppl_3 = []
            ppl_4 = []
            for text in (load_sentences('{}{}'.format(data_path, file))[:10]):
                encoded_tokens = [eos_token]
                encoded_tokens += enc.encode(text.strip())
                num_tokens = len(encoded_tokens)
                num_words = len(text.split(' '))
                log_probs = sess.run(probs, feed_dict={
                    tokens: [encoded_tokens for _ in range(batch_size)]})

                nll = []
                for idx, token_logit in enumerate(log_probs):
                    # Build NLL loss for PPL
                    nll.append(math.log(token_logit[encoded_tokens[idx]], 2))

                ppl_1.append(2 ** (-1 * np.mean(nll)))  # norm by #tokens
                ppl_2.append(2 ** (-1 * (np.sum(nll)/(num_tokens-1))))  # norm by #tokens-1
                ppl_3.append(2 ** (-1 * (np.sum(nll)/num_words)))  # norm by #words

            ppl_1 = np.mean(ppl_1)
            ppl_2 = np.mean(ppl_2)
            ppl_3 = np.mean(ppl_3)
            print('PPL for {}:\t{:.4f} | {:.4f} | {:.4f}'.format(
                file, ppl_1, ppl_2, ppl_3))
            corpus_ppls[file] = [ppl_1, ppl_2, ppl_3]
    return corpus_ppls


if __name__ == '__main__':
    with open('corpus_ppls_1.pkl', 'wb') as f:
        pickle.dump(evaluate_model(), f)
