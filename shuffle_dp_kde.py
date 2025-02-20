import sys
import os
import re
from importlib import import_module

from datasets import load_dataset

from sentence_transformers import SentenceTransformer

import numpy as np
import scipy.sparse as sp
import torch
import pickle

# Settings

FILE_PATH = "./data/"

SBERT_MODEL_NAME = "all-mpnet-base-v2"


# Textual datasets

dic_ag_news = {"name": "ag_news", "n_categories": 4, "text_field": "text", "test_split": "test"}
dic_sst2 = {"name": "sst2", "n_categories": 2, "text_field": "sentence", "test_split": "validation"}
dic_dbpedia_14 = {"name": "dbpedia_14", "n_categories": 14, "text_field": "content", "test_split": "test"}

DATASET_DIC = {"ag_news": dic_ag_news,
               "sst2": dic_sst2,
               "dbpedia_14": dic_dbpedia_14}


### Embed dataset in R^d ###

def process_and_save_data_embedding(dataset_name, save_file=False):
    print("Processing embeddings for dataset:", dataset_name)
    dataset_settings = DATASET_DIC[dataset_name]
    train_set = load_dataset(dataset_settings["name"], split="train")
    test_set = load_dataset(dataset_settings["name"], split=dataset_settings["test_split"])
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    train_classes = []
    translated_data = []
    translated_queries = []
    for c in range(dataset_settings["n_categories"]):
        train_classes.append([])
    record_count = 0
    for record in train_set:
        if record_count < 10 or record_count % 100 == 0:
            print("Record count:", record_count)
        record_count += 1
        record_text = record[dataset_settings["text_field"]]
        processed = sbert_model.encode(record_text)
        train_classes[record["label"]].append(processed)
    nq = len(test_set)
    processed_queries = []
    query_labels = np.zeros(nq)
    for r_idx, record in enumerate(test_set):
        if r_idx < 10 or r_idx % 100 == 0:
            print("Record count:", r_idx)
        record_text = record[dataset_settings["text_field"]]
        processed_queries.append(sbert_model.encode(record_text))
        query_labels[r_idx] = record["label"]
    if save_file:
        file_name = os.path.join(FILE_PATH, "emb_" + dataset_name + ".npy")
        np.save(file_name, [train_classes, np.vstack(processed_queries), query_labels, dataset_settings["sbert_model_name"]])
        print("Saved file:", file_name)
    return train_classes, np.vstack(processed_queries), query_labels


### Shuffled DP KDE ###

class ShuffleDensity:

    def __init__(self, dataset, total_n, rff_reps=0, delta=1./100000, bandwidth=1):
    
        # By convention: if rff_reps==0 use the IP kernel, if rff_reps > 0 use the Gaussian kernel with rff_reps many random Fourier features
    
        self.n = dataset.shape[0]
        self.d = dataset.shape[1]
        self.crd_range = 1

        self.use_ip_kernel = rff_reps == 0
        
        if self.use_ip_kernel:
            self.reps = self.d
        else:
            self.reps = rff_reps # number of RFFs
            self.rff = np.sqrt(2. / bandwidth) * np.random.normal(0, 1, (self.d, self.reps))
            self.rff_shift = np.random.uniform(0, 2*np.pi, self.reps).reshape(1, -1)
        self.query_fnc = lambda data_sketch, query_sketch: (1./self.reps) * np.dot(data_sketch, query_sketch.T)

        # Params for shuffled protocols and composition
        self.delta0 = 0.5 * delta
        self.delta_prime_composition = 0.5 * delta
        self.total_delta = self.delta0 + self.delta_prime_composition

        # Sketch dataset
        self.process_dataset(dataset)

    def process_features(self, m):
        if self.use_ip_kernel:
            output = m
        else:
            output = np.cos(np.dot(m, self.rff) + self.rff_shift)
        return output

    def process_data_features(self, m):
        return self.process_features(m)

    def process_query_features(self, m):
        return self.process_features(m)

    def process_dataset(self, dataset):
        # Prepare dataset features
        self.processed_dataset = self.process_data_features(dataset)
        self.unsanitized_mean = np.mean(self.processed_dataset, axis=0)

        # Quantize dataset features
        rounding_probability_matrix = 0.5 * (1 - self.processed_dataset * 1. / self.crd_range)
        q_dataset = np.zeros(self.processed_dataset.shape)
        q_dataset += self.processed_dataset
        q_dataset[np.random.uniform(0, 1, q_dataset.shape) < rounding_probability_matrix] = -self.crd_range
        q_dataset[q_dataset > -self.crd_range] = self.crd_range
        self.quantized_processed_dataset = q_dataset
        self.quantized_unsanitized_mean = np.mean(self.quantized_processed_dataset, axis=0)
        self.bit_quantized_processed_dataset = (self.quantized_processed_dataset + self.crd_range) * 1. / (2 * self.crd_range)

    def sanitize_centrally(self, epsilon, approx=True):
        self.cdp_sanitized_dataset = np.mean(self.processed_dataset, axis=0)
        if approx:
            std = self.crd_range * np.sqrt(2 * np.log(1.25 / self.total_delta) * self.reps) / (epsilon * self.n)
            self.cdp_sanitized_dataset += np.random.normal(0, std , self.reps)
            return epsilon, self.total_delta
        else:
            laplace_param = self.crd_range * self.reps * 2. / (epsilon * self.n)
            self.cdp_sanitized_dataset += np.random.laplace(0, laplace_param, self.reps)
            return epsilon, 0

    def sanitize_locally_rr(self, flip_probability):
        flips = np.zeros(self.quantized_processed_dataset.shape)
        flips[np.random.uniform(0, 1, flips.shape) < flip_probability] = 1
        sanitized_quantized_processed_dataset = self.quantized_processed_dataset * (-1) ** flips
        # Let ~f be f with each entry flipped w.p. p from 1 to -1 or vice versa, where p=flip_probability
        # then E[~f] = (1-2p)f, hence E[<g,~f>] = (1-2p)<g,f>, hence below we renormalize by 1/(1-2p)
        self.locally_sanitized_dataset_rr = (1. / (1 - 2 * flip_probability)) * np.mean(sanitized_quantized_processed_dataset, axis=0)

    def get_epsilon_delta_to_use(self, epsilon):
        epsilon_to_use = epsilon * 1. / self.reps
        delta_to_use = self.delta0  * 1. / self.reps
        return epsilon_to_use, delta_to_use
    
    def sanitize_for_shuffle_rr(self, epsilon):
        epsilon_to_use, delta_to_use = self.get_epsilon_delta_to_use(epsilon)

        # Set the parameter lambda from Cheu et al., Lemma 4.8
        if epsilon_to_use >= (192. * np.log(4. / delta_to_use) / self.n) ** 0.5:
            # Case 1 of Lemma 4.8
            cheu_lambda = 64. * np.log(4. / delta_to_use) / (epsilon_to_use ** 2)
        else:
            # Case 2 of Lemma 4.8
            cheu_lambda = self.n - epsilon_to_use * (self.n ** 1.5) / (432. * np.log(4. / delta_to_use)) ** 0.5
        flip_probability = cheu_lambda * 1. / (2 * self.n)
        
        # Sanitize
        self.sanitize_locally_rr(flip_probability)
        self.shuffle_sanitized_dataset = self.locally_sanitized_dataset_rr
        return self.get_composition_epsilon_delta(epsilon_to_use, delta_to_use)

    def set_shuffle_sanitized_dataset_from_bitsums(self, bitsums):
        self.shuffle_sanitized_dataset = 2 * self.crd_range * (bitsums - 0.5)

    def sanitize_d_shuffle(self, piece_distribution, distribution_mean):
        # Randomizer adds noise from distribution piece
        randomizer_output = self.bit_quantized_processed_dataset + piece_distribution(self.bit_quantized_processed_dataset.shape)
        # Analyzer computes sum shifted by distribution mean
        analyzer_output = (1. / self.n) * (np.sum(randomizer_output, axis=0) - distribution_mean)
        self.set_shuffle_sanitized_dataset_from_bitsums(analyzer_output)

    def sanitize_d3_shuffle(self, piece_distribution1, piece_distribution2, distribution_mean1, distribution_mean2):
        # For simulation purposes, the third distribution cancels out so we don't need to simulate it; 
        # then the mechanism reduces to the D-distributed mechanism with D=D1-D2
        self.sanitize_d_shuffle(lambda size:piece_distribution1(size) - piece_distribution2(size), distribution_mean1 - distribution_mean2)

    def sanitize_3negbin(self, epsilon):
        # The correlated noise mechanism from Ghazi et al. ICML 2020
        epsilon_to_use, delta_to_use = self.get_epsilon_delta_to_use(epsilon)
        negbin_param_r = 1.
        try:
            negbin_param_p = np.e ** (-0.8 * epsilon_to_use)
        except:
            negbin_param_p = 1
        # numpy.random parameterizes the negative binomial distribution differently from Ghazi et al. ICML 2020, so, flip p:
        negbin_param_p = 1 - negbin_param_p 
        negbin_piece_distribution = lambda size:np.random.negative_binomial(negbin_param_r / self.n, negbin_param_p, size)
        negbin_mean = 0 # The means zero out
        self.sanitize_d3_shuffle(negbin_piece_distribution, negbin_piece_distribution, negbin_mean, negbin_mean)
        return self.get_composition_epsilon_delta(epsilon_to_use, delta_to_use)

    def sanitize_pure_shuffle(self, epsilon, param_rho=0.5):
        # The pure DP mechanism from Ghazi, Kumar, Manurangsi 2023
        # param_rho from their paper controls how close error would be to pure central DP, and is in (0.0.5]
        epsilon_to_use, delta_to_use = self.get_epsilon_delta_to_use(epsilon)
        # Randomizer zeros each input bit (i.e., fails to send it) w.p. q. Otherwise, sends it. The parameter s cancels out.
        # Set parameters as per section 4.4 of paper
        param_eps_prime = epsilon_to_use - 0.01 * param_rho * min(epsilon_to_use, 1)
        param_q = 0.2 * param_rho / ((np.e ** epsilon_to_use) * ((1 - np.e ** (-epsilon_to_use)) ** 2) * self.n)
        # Simulate
        shape = self.bit_quantized_processed_dataset.shape
        randomized_input = np.ones(shape)
        randomized_input[np.random.uniform(0, 1, shape) < param_q] = 0
        randomized_input = randomized_input * self.bit_quantized_processed_dataset
        # Noise messages (note: flood messages cancel out and need not be simulated)
        # The noise distribution is Geo(1-e^{-epsilon'}) which is a.k.a. NegBin(1, 1-e^{-epsilon'}), hence piece distribution is:
        noise_piece_distribution = lambda size:np.random.negative_binomial(1. / self.n, 1 - np.e ** (-param_eps_prime), size)
        noise_matrix = noise_piece_distribution(shape) - noise_piece_distribution(shape)
        # Analyzer 
        analyzer_output = np.mean(randomized_input + noise_matrix, axis=0)
        # Finish and perform composition
        self.set_shuffle_sanitized_dataset_from_bitsums(analyzer_output)
        return self.get_pure_composition_epsilon(epsilon_to_use), 0
        
    def get_composition_epsilon_delta(self, input_epsilon, input_delta, pure_mechanisms=False):
        if pure_mechanisms:
            # All composed mechanisms are pure, can use full delta "budget" for composition
            delta_prime_composition = self.total_delta
        else:
            delta_prime_composition = self.delta_prime_composition
        # Compute and return actual privacy parameters according to composition theorem
        epsilon_final = input_epsilon * (np.e ** input_epsilon - 1) * self.reps + \
                        input_epsilon * (2 * self.reps * np.log(1. / delta_prime_composition)) ** 0.5
        delta_final = delta_prime_composition + input_delta * self.reps
        return epsilon_final, delta_final

    def get_pure_composition_epsilon(self, input_epsilon):
        return input_epsilon * self.reps

    def non_private_kde(self, queries):
        return self.query_fnc(self.unsanitized_mean, self.process_query_features(queries))

    def centrally_private_kde(self, queries):
        return self.query_fnc(self.cdp_sanitized_dataset, self.process_query_features(queries))

    def shuffle_private_kde(self, queries):
        return self.query_fnc(self.shuffle_sanitized_dataset, self.process_query_features(queries))


# Protect labels with local DP
def label_rr(train_classes, epsilon_rr):
    nc = len(train_classes)
    new_train_classes = [[] for _ in range(nc)]
    for cidx in range(nc):
        for datapoint in train_classes[cidx]:
            if np.random.uniform(0, 1) < (np.e ** epsilon_rr - 1) * 1. / (np.e ** epsilon_rr - 1 + nc):
                new_train_classes[cidx].append(datapoint)
            else:
                new_train_classes[np.random.randint(nc)].append(datapoint)
    return new_train_classes


# Test

def load_data(dataset_name):
    dataset_settings = DATASET_DIC[dataset_name]
    file_name = os.path.join(FILE_PATH, "emb_" + dataset_name) + ".npy"
    saved_data = np.load(file_name, allow_pickle=True)
    train_classes = saved_data[0]
    queries = saved_data[1]
    query_labels = saved_data[2]
    
    # Normalize embeddings
    for cidx in range(len(train_classes)):
        train_classes[cidx] /= np.linalg.norm(train_classes[cidx], axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    
    return train_classes, queries, query_labels
        

def test_shuffled_dp_kde(train_classes, queries, query_labels, privacy="none", epsilon=1, reps=0, bandwidth=1, label_rr_eps=0, delta=1./100000, do_class_decoding=True):
    # By convention: if reps==0 use IP kernel, if reps>0 use Gaussian kernel with reps many random Fourier features
    # By convention: if label_rr_eps==0 do not protect labels, if label_rr_eps>0 protect labels with (label_rr_eps,0)-local DP

    dataset_settings = DATASET_DIC[dataset_name]
    total_n = sum([len(ccc) for ccc in train_classes])
    
    if privacy == "none":
        print("Method: No privacy")
    elif privacy == "central":
        print("Method: Central DP")
    elif privacy == "rr":
        print("Method: Shuffled DP with RR bitsum" )
    elif privacy == "3nb":
        print("Method: Shuffled DP with 3NB bitsum" )
    elif privacy == "pure":
        print("Method: Shuffled DP with Pure bitsum" )
    else:
        print("Unknown privacy setting")
        return
    print("Dataset name:", dataset_name)
    print("Categories:", dataset_settings["n_categories"])

    if label_rr_eps == 0:
        print("Using labels in the clear")
    else:
        print("Doing label RR with epsilon", label_rr_eps)
        train_classes = label_rr(train_classes, label_rr_eps)

    # Sketch data
    print("Sketching data...")
    dpkde_list = []
    for c, class_dataset in enumerate(train_classes):
        print("Sketching class", c)
        lsq_rff = ShuffleDensity(np.vstack(class_dataset), total_n, reps, delta, bandwidth)
        final_epsilon = 0
        final_delta = 0
        ldp_eps = 0
        # Sanitize
        if privacy == "central":
            final_epsilon, final_delta = lsq_rff.sanitize_centrally(epsilon)
        elif privacy == "rr":
            final_epsilon, final_delta = lsq_rff.sanitize_for_shuffle_rr(epsilon)
        elif privacy == "3nb":
            final_epsilon, final_delta = lsq_rff.sanitize_3negbin(epsilon)
        elif privacy == "pure":
            final_epsilon, final_delta = lsq_rff.sanitize_pure_shuffle(epsilon, True)
        if c == 0:
            print("Final epsilon:", final_epsilon)
            print("Final delta:", final_delta)
        dpkde_list.append(lsq_rff)
    print("Done sketching data")

    print("Measuring accuracy...")
    # Measure accuracy
    nq = len(query_labels)
    class_scores = np.zeros((nq, len(dpkde_list)))
    for i in range(len(dpkde_list)):
        print(i, "...")
        if privacy == "none":
            class_scores[:, i] = dpkde_list[i].non_private_kde(queries)
        elif privacy == "central":
            class_scores[:, i] = dpkde_list[i].centrally_private_kde(queries)
        elif privacy in ["rr", "3nb", "pure"]:
            class_scores[:, i] = dpkde_list[i].shuffle_private_kde(queries)
    predictions = class_scores.argmax(axis=1)
    success_rate = np.sum(predictions == query_labels) * 1. / nq
    print("Done measuring accuracy")
    print("Total queries:", nq)
    print("Success rate:", success_rate)
    ### Class decoding
    if not do_class_decoding:
        return success_rate, final_epsilon, []
    print("Doing class decoding")
    queries = np.load(os.path.join(FILE_PATH, "glove50emb.npy"), allow_pickle=True)
    vocab = eval(open(os.path.join(FILE_PATH, "glove50embwords.npy")).read())
    class_decoding_output = []
    for i in range(len(dpkde_list)):
        print(i, "...")
        if privacy == "none":
            class_scores = dpkde_list[i].non_private_kde(queries)
        elif privacy == "central":
            class_scores = dpkde_list[i].centrally_private_kde(queries)
        elif privacy in ["rr", "3nb", "pure"]:
            class_scores = dpkde_list[i].shuffle_private_kde(queries)
        best_ind = class_scores.argsort()[::-1][:20]
        class_decoding_output.append([vocab[ind] for ind in best_ind])
        print(class_decoding_output[-1])       
    return success_rate, final_epsilon, class_decoding_output


### USAGE EXAMPLE ###

# First, create and save data embeddings by running: process_and_save_data_embedding("ag_news", True)
# Or "dbpedia_14" or "sst2" instead of "ag_news"
# Once embeddings are saved, run protocol.
# Instead, to create data embeddings on the fly, set LOAD_FROM_FILE below to False.

dataset_name = sys.argv[1]
reps = int(sys.argv[2]) # 0 for IP kernel, positive number for Gaussian kernel
epsilon = float(sys.argv[3])
epsilon_label = float(sys.argv[4]) # 0 for no label protection
delta=1./100000
bitsum = sys.argv[5] # options: "none", "central", "rr", "3nb", "pure"

# NOTE:
# For "none", epsilon is ignored (this is a no-privacy setting).
# For "central" and "pure", the protocol will be (epsilon,delta)-shuffled DP.
# For "rr" and "3nb", due to composition, the output will be (final_epsilon,delta)-shuffled DP, where final_epsilon is
# a value smaller than epsilon, possibly by a lot, that will be returned by the protocol, based on advanced composition.
# Thus, the input epsilon parameter is a (possibly very loose) upper bound on the actual privacy parameter final_epsilon.

LOAD_FROM_FILE = True


print("Loading data...")
if LOAD_FROM_FILE:
    train_classes, queries, query_labels = load_data(dataset_name)
else:
    train_classes, queries, query_labels = process_and_save_data_embedding("ag_news", False)

print("Executing protocol...")
success_rate, final_epsilon, class_decoding_results = test_shuffled_dp_kde(train_classes, queries, query_labels, bitsum, epsilon, reps, 1, epsilon_label, delta, False)
print("Done")
