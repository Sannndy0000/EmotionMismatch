import liwc
from collections import Counter
import itertools
from nltk import word_tokenize
from textstat.textstat import textstat
import networkx as nx
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import numpy as np
import truecase
import re
import gensim
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import truecase

parse, category_names= liwc.load_token_parser('/nfs/turbo/McInnisLab/sandymn/SI-brown/text_feats/LIWC2015_English.dic')




#############################  LIWC  ################################
def extract_liwc(df_c, feats):
    text_col = 'transcriptions'
    df_c = df_c.fillna(value={'transcriptions':' '})
    text = ' '.join(df_c[text_col]).replace('<comma>', ',').replace('<filler>', ' ')
    # feats['transcriptions'] = text
    tokens = word_tokenize(text)
    if len(tokens) == 0:
        for c in category_names:
            feats['TEXT_liwc.'+ c] = float('nan')
    else:
        counts = Counter(category for token in tokens for category in parse(token))
        for c in category_names:
            feats['TEXT_liwc.'+ c] = counts.get(c, 0)/len(tokens)
    
    return feats


###########################  verbosity  ##############################
def extract_verbosity(df_c, feats):
    text_col = 'transcriptions'
    word_count_list = []
    word_lengths = []
    long_count = 0
    df_c = df_c.fillna(value={'transcriptions':' '})
    for i, row in df_c.iterrows():
        segment = word_tokenize(row[text_col].replace('<comma>', ',').replace('<filler>', ' '))
        word_count_list.append(len(segment))
        for word in segment:
            word_lengths.append(len(word))
            if len(word) > 6:
                long_count += 1
    # Compute segment level statistics
    feats['TEXT_wc'] = np.sum(word_count_list)
    feats['TEXT_wc_mean'] = np.mean(word_count_list) if word_count_list else float('nan')
    feats['TEXT_wc_median'] = np.median(word_count_list) if word_count_list else float('nan')
    feats['TEXT_wc_stdev'] = np.std(word_count_list) if word_count_list else float('nan')
    feats['TEXT_wc_min'] = min(word_count_list) if word_count_list else float('nan')
    feats['TEXT_wc_max'] = max(word_count_list) if word_count_list else float('nan')
    feats['TEXT_total_count'] = sum(word_count_list) if word_count_list else float('nan')

    # Compute fraction of words across whole call that are long (i.e. 6+ words)
    feats['TEXT_lw_count'] = (long_count / feats['TEXT_total_count']) if feats['TEXT_total_count'] else float('nan')
    # Compute mean length of any word used
    feats['TEXT_word_len'] = np.mean(word_lengths) if word_lengths else float('nan')
    
    return feats


def extract_syll_stats(df_c, feats):
    df_c = df_c.fillna(value={'transcriptions':' '})
    segments = [word_tokenize(x.replace('<comma>', ',').replace('<filler>', ' ')) for x in list(df_c['transcriptions'])]
    syll_count_list = []
    for segment in segments:
        for word in segment:
            syll_count_list.append(textstat.syllable_count(word))
    feats['TEXT_syll_mean'] = np.mean(syll_count_list) if syll_count_list else float('nan')
    feats['TEXT_syll_median'] = np.median(syll_count_list) if syll_count_list else float('nan')
    feats['TEXT_syll_stdev'] = np.std(syll_count_list) if syll_count_list else float('nan')
    feats['TEXT_syll_min'] = min(syll_count_list) if syll_count_list else float('nan')
    feats['TEXT_syll_max'] = max(syll_count_list) if syll_count_list else float('nan')
    
    return feats


###########################  diversity  ##############################

def compute_MATTR(words, feats, window):
    """
    Computes MATTR for given window size.
    :param words: list of all words in transcript
    :param feats: dictionary to store feature values for the transcript
    :param window: size of window to compute each TTR over.
    If window size is larger than number of words across all segments, uses total number of words as window size
    and prints warning message.
    """
    original_window = window
    if len(words) == 0:
        # if transcript is empty, MATTR is not defined
        feats["TEXT_MATTR_{}".format(original_window)] = float('nan')
        return
    if len(words) < window:
        # print("WARNING: window size {} greater than number of words in transcript {}. Using window size {}.".format(
        #     window, len(words), len(words)))
        window = len(words)
    # store counts of words in current window
    vocab_dict = {}
    # add words from first window
    for i in range(window):
        word = words[i]
        if word not in vocab_dict:
            vocab_dict[word] = 0
        vocab_dict[word] += 1
    # store list of TTR for each window
    ttrs = [len(vocab_dict.keys())/float(window)]
    # keep track of first word in window (will be the one word not also in next window)
    first_word = words[0]
    for i in range(1, len(words) - window + 1):
        # remove instance of first word in previous window
        vocab_dict[first_word] -= 1
        if vocab_dict[first_word] == 0:
            del vocab_dict[first_word]
        first_word = words[i]
        # add word that wasn't in previous window (last word in this window)
        last_word = words[i + window - 1]
        if last_word not in vocab_dict:
            vocab_dict[last_word] = 0
        vocab_dict[last_word] += 1
        ttrs.append(len(vocab_dict.keys())/float(window))
    feats["TEXT_MATTR_{}".format(original_window)] = np.mean(ttrs)


def get_honores_statistic(words, feats):
    """
    Computes Honore's Statistic, a measure which emphasizes the use of words that are only used once.
    :param words: list of all words in transcript
    :param feats: dictionary to store feature values for the transcript
    """
    total_words = len(words)
    unique_words = len(set(words))
    single_time_words = len([word for word in words if words.count(word) == 1])
    if total_words == 0:
        feats["TEXT_HS"] = float('nan')
        return
    # smooth statistic so not undefined when # unique words = # single time words
    epsilon = 1e-5
    feats["TEXT_HS"] = 100 * np.log(total_words / float(1 - single_time_words / float(unique_words + epsilon)))


def extract_diversity(df_c, feats):
    """
    Computes lexical diversity features for input text document and stores in dictionary.
    :param text: string with the text of a document to extract features for
    :return: Dictionary mapping feature names to values
    """
    # collect all words
    text = ' '.join(df_c['transcriptions']).replace('<comma>', ',').replace('<filler>', ' ')
    words = word_tokenize(text)
    feats = {}
    for window in [10, 25, 50]:
        compute_MATTR(words, feats, window)
    get_honores_statistic(words, feats)
    return feats


###########################  POS  ##############################
POS_KEY_LIST = ['ADJ', 'VERB', 'NOUN', 'ADV', 'DET', 'INT', 'PREP', 'CC', 'PNOUN', 'PSNOUN']


def update_feature_vals(tag, feats):
    """
    Updates feature values (POS counts) based on input POS tag.
    :param tag: Penn TreeBank tag
    :param feats: dictionary to store feature values for the transcript
    """
    if tag.startswith('J'):
        feats['TEXT_ADJ'] += 1
    elif tag.startswith('V'):
        feats['TEXT_VERB'] += 1
    elif tag.startswith('N'):
        feats['TEXT_NOUN'] += 1
    elif tag.startswith('R'):
        feats['TEXT_ADV'] += 1
    elif tag.startswith('D'):
        feats['TEXT_DET'] += 1
    elif tag.startswith('U'):
        feats['TEXT_INT'] += 1
    elif tag.startswith('I') or tag.startswith('T'):
        feats['TEXT_PREP'] += 1
    elif tag == 'CC':
        feats['TEXT_CC'] += 1
    elif tag == 'PRP':
        feats['TEXT_NOUN'] += 1
        feats['TEXT_PNOUN'] += 1
    elif tag == 'PRP$':
        feats['TEXT_PSNOUN'] += 1
        feats['TEXT_NOUN'] += 1
    elif tag.startswith('W'):
        if tag[1] == 'D':
            feats['TEXT_DET'] += 1
        elif tag[1] == 'R':
            feats['TEXT_ADV'] += 1
        elif tag.endswith('P'):
            feats['TEXT_PNOUN'] += 1
            feats['TEXT_NOUN'] += 1
        else:
            feats['TEXT_PSNOUN'] += 1


def get_pos_ratios(pos_dict):
    pos_dict['TEXT_adj_ratio'] = float(pos_dict['TEXT_ADJ']) / float(pos_dict['TEXT_VERB']) \
        if float(pos_dict['TEXT_VERB']) else float('nan')
    pos_dict['TEXT_v_ratio'] = float(pos_dict['TEXT_NOUN']) / float(pos_dict['TEXT_VERB']) if float(pos_dict['TEXT_VERB']) else float('nan')
    if float(pos_dict['TEXT_VERB']) + float(pos_dict['TEXT_NOUN']):
        pos_dict['TEXT_n_ratio'] = float(pos_dict['TEXT_NOUN']) / (float(pos_dict['TEXT_VERB']) + float(pos_dict['TEXT_NOUN']))
    else:
        pos_dict['TEXT_n_ratio'] = float('nan')
    pos_dict['TEXT_pn_ratio'] = float(pos_dict['TEXT_PNOUN']) / float(pos_dict['TEXT_NOUN']) \
        if float(pos_dict['TEXT_NOUN']) else float('nan')
    pos_dict['TEXT_sc_ratio'] = float(pos_dict['TEXT_PREP']) / float(pos_dict['TEXT_CC']) if float(pos_dict['TEXT_CC']) else float('nan')


def extract_pos(df_c, feats):
    """
    :param segments: List of text segments. Each segment is a string.
    :return: feats: Dictionary mapping feature name to value for transcript
    Note:  POS is more accurate for segments that contain capitalization, so if text is fully lowercase
           this function will try to infer the true casing before POS detection.
    """
    # initialize feature dictionary with POS types
    feats.update(dict(('TEXT_'+key, 0) for key in POS_KEY_LIST))
    num_words = 0
    # add POS count features
    segments = list(df_c['transcriptions'])
    for segment in segments:
        lowercase = segment.islower()
        # split up into words
        segment = segment.split(" ")
        num_words += len(segment)
        if lowercase:
            # if text has been lowercased,
            # transform to true case (i.e. capitalize if supposed to be), so that POS tagger works better
            segment_str = " ".join(segment)
            truecase_str = truecase.get_true_case(segment_str)
            segment = truecase_str.split(" ")
        if '' in set(segment):
            segment[:] = [w for w in segment if w != '']
        pos_seg = nltk.pos_tag(segment)
        for word, tag in pos_seg:
            update_feature_vals(tag, feats)
    get_pos_ratios(feats)
    # convert counts to proportions
    for key in POS_KEY_LIST:
        count = float(feats['TEXT_'+key])
        feats['TEXT_'+key] = count / float(num_words)
    return feats


def extract_timing(df_c, feats):
    df_c = df_c.fillna(value={'transcriptions':' '})
    segments = [word_tokenize(x) for x in list(df_c['transcriptions'])]
    word_time_list = []
    df_c = df_c.reset_index()
    for i, row in df_c.iterrows():
        word_time_list.append(row['segment_seconds']/len(segments[i]) if len(segments[i])>0 else float('nan'))
    seg_time_list = list(df_c['segment_seconds'])
    
    feats['TEXT_wtime_mean'] = np.mean(word_time_list) if word_time_list else float('nan')
    feats['TEXT_wtime_median'] = np.median(word_time_list) if word_time_list else float('nan')
    feats['TEXT_wtime_stdev'] = np.std(word_time_list) if word_time_list else float('nan')
    feats['TEXT_wtime_min'] = min(word_time_list) if word_time_list else float('nan')
    feats['TEXT_wtime_max'] = max(word_time_list) if word_time_list else float('nan')
    feats['TEXT_stime_mean'] = np.mean(seg_time_list) if seg_time_list else float('nan')
    feats['TEXT_stime_median'] = np.median(seg_time_list) if seg_time_list else float('nan')
    feats['TEXT_stime_stdev'] = np.std(seg_time_list) if seg_time_list else float('nan')
    feats['TEXT_stime_min'] = min(seg_time_list) if seg_time_list else float('nan')
    feats['TEXT_stime_max'] = max(seg_time_list) if seg_time_list else float('nan')
    
    return feats


def extract_confi(df_c, feats):
    confi_list = list(df_c['confidence'])
    feats['TEXT_confi_mean'] = np.mean(confi_list) if confi_list else float('nan')
    feats['TEXT_confi_median'] = np.median(confi_list) if confi_list else float('nan')
    feats['TEXT_confi_stdev'] = np.std(confi_list) if confi_list else float('nan')
    feats['TEXT_confi_min'] = min(confi_list) if confi_list else float('nan')
    feats['TEXT_confi_max'] = max(confi_list) if confi_list else float('nan')
    
    return feats



# below are functions for graph features
LEMMATIZER = WordNetLemmatizer()
nltk.download('punkt')
SENTENCE_TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')


def get_wordnet_pos(treebank_tag):
    """
    :param treebank_tag: Penn Treebank POS tag
    :return: corresponding Word Net tag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize(segments):
    """
    Converts speech segments from list of words to list of their associated lemmas.
    :param segments: a list of speech segments, where each segment is a list of the words that make up that segment
    :return: segments: speech segment list with words converted to their associated lemma
    """
    for segment in segments:
        pos_list = nltk.pos_tag(segment)
        for i in range(len(segment)):
            pos = get_wordnet_pos(pos_list[i][1])
            segment[i] = LEMMATIZER.lemmatize(segment[i].lower(), pos)  # make segment lowercase before lemmatizing
    return segments


def create_naive_graph(segments):
    """
    creates a naive graph representation from the set of text segments
    :param segments: a list of segments, where each segment is a list of the words that make up that segment
    :return: g: a graph made up of nodes representing each distinct word used throughout all the segments
    and edges that link consecutive words (words are only considered consecutive if they are adjacent within the
    same segment). The graph is in the form of a directed multigraph (edges have direction and the graph can have
    self-loops and parallel edges).
    """
    g = nx.MultiDiGraph()
    for segment in segments:
        for i in range(len(segment) - 1):
            g.add_edge(segment[i], segment[i + 1])
        if len(segment) == 1:
            g.add_node(segment[0])
    return g


def create_lemma_graph(segments):
    """
    creates a lemma graph representation from the set of text segments
    :param segments: a list of segments, where each segment is a list of the words that make up that segment
    :return: g: a graph made up of nodes representing each distinct lemma word used throughout all the segements and
    edges that link consecutive words (in the form of their lemmas). Again, words are only considered consecutive if
    they are adjacent within the same segment.
    """
    segments = lemmatize(segments)
    g = create_naive_graph(segments)
    return g


def create_pos_graph(segments):
    """
    creates a part of speech graph representation from the set of text segments
    :param segments: a list of segments, where each segment is a list of the words that make up that segment
    :return: g: a graph made up of nodes representing each distinct part of speech used throughout all the segments and
    edges that link parts of speech that are used in consecutively (adjacent within the same sentence)
    """
    for i in range(len(segments)):
        # transform words to their associated parts of speech
        segments[i] = nltk.pos_tag(segments[i])
        for j in range(len(segments[i])):
            segments[i][j] = segments[i][j][1]
    g = create_naive_graph(segments)
    return g


def get_connectivity_measures(graph, u_graph, graph_type, feats):
    """
    Computes graph measures related to connectivity: average node degree (ATD), largest connected component (LCC),
    and largest strongly connected component (LSC).
    :param graph: a directed multigraph (i.e. can have self loops and parallel edges)
    :param u_graph: an undirected multigraph
    :param graph_type: naive, lemma, or POS
    :param feats: dictionary to store feature values for the transcript
    """
    # calculate average degree of every node in the graph (ATD)
    atd = 0
    node_list = graph.nodes()
    for node in node_list:
        degree = graph.degree(node)
        atd += degree
    if len(graph):
        atd /= len(graph)
    else:
        atd = float('nan')
    feats['TEXT_ave_degree_{}'.format(graph_type)] = atd

    # calculate number of nodes in maximum connected component(LCC)
    components = sorted(nx.connected_components(u_graph), key=len, reverse=True)
    if components:
        feats['TEXT_lcc_{}'.format(graph_type)] = len(components[0])
    else:
        feats['TEXT_lcc_{}'.format(graph_type)] = 0

    # calculate number of nodes in maximum strongly connected component (LSC)
    s_components = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    if s_components:
        feats['TEXT_lsc_{}'.format(graph_type)] = len(s_components[0])
    else:
        feats['TEXT_lsc_{}'.format(graph_type)] = 0


def get_parallel_edges(graph, graph_type, feats):
    """
    Calculate number of parallel edges in graph. Edges must be in same direction to count as parallel
    (L2 measure counts loops with two nodes). Each repeated edge count as one parallel edge.
    Store parallel edge count in feats.
    In density metric, compute E' = E - (L1 + PE). However, self-loops (L1) can be parallel edges, and they shouldn't
    be double counted in E' measure. Therefore, also returns count of edges that are self-loops and parallel.
    :param graph: a directed multigraph (i.e. can have self loops and parallel edges)
    :param graph_type: naive, lemma, or POS
    :param feats: dictionary to store feature values for the transcript
    :return: num_p_edges: total count of parallel edges,
    pe_l1_count: count of edges that are both parallel and self-loops
    """
    num_p_edges = 0
    edge_list = list(graph.edges())
    edge_set = set(edge_list)
    pe_l1_count = 0
    for edge in edge_set:
        occurrences = edge_list.count(edge)
        if occurrences > 1:
            if edge[0] == edge[1]:
                pe_l1_count += (occurrences - 1)
            num_p_edges += (occurrences - 1)
    feats['TEXT_num_p_edges_{}'.format(graph_type)] = num_p_edges
    return num_p_edges, pe_l1_count


def get_loops(graph, graph_type, feats):
    """
    Calculate number of loops with two nodes (L2) and with three nodes (L3).
    :param graph: a directed multigraph (i.e. can have self loops and parallel edges)
    :param graph_type: naive, lemma, or POS
    :param feats: dictionary to store feature values for the transcript
    """
    adj_mat = nx.to_numpy_matrix(graph)
    # make sure self loops aren't counted
    # otherwise traversing a self-loop two/three times in a row will be counted as a loop with two/three nodes
    np.fill_diagonal(adj_mat, 0)
    squared_mat = np.matmul(adj_mat, adj_mat)
    # divide by two/three because loop is counted once for each node in loop in sum of trace
    feats['TEXT_l2_{}'.format(graph_type)] = np.trace(squared_mat) / 2
    cubed_mat = np.matmul(adj_mat, squared_mat)
    feats['TEXT_l3_{}'.format(graph_type)] = np.trace(cubed_mat) / 3


def get_shortest_path_metrics(u_graph, graph_type, feats):
    """
    Compute graph measures related to shortest path lengths: diameter (DI: longest shortest path between any two nodes),
    average shortest path (ASP: average length of the shortest path between pairs of nodes in a graph, computed across
    all connected components)
    Measures are computed using undirected graph following paper by Mota et al.
    :param u_graph: an undirected multigraph (loops and parallel edges don't affect computation as the shortest
    path won't include them)
    :param graph_type: naive, lemma, or POS
    :param feats: dictionary to store feature values for the transcript
    """
    longest = 0
    average = 0
    num_pairs = 0
    #for component in nx.connected_component_subgraphs(u_graph):
    cc_subgraphs = (u_graph.subgraph(c) for c in nx.connected_components(u_graph)) #networkx v2.5
    for component in cc_subgraphs:
        lengths = dict(nx.all_pairs_shortest_path_length(component))
        nodes = list(component.nodes())
        num_nodes = len(nodes)
        num_pairs += float((num_nodes * (num_nodes - 1) / 2))
        for i in range(num_nodes):
            # don't check for shortest path between node and itself
            for j in range(i + 1, num_nodes):
                path_length = lengths[nodes[i]][nodes[j]]
                if path_length > longest:
                    longest = path_length
                average += path_length
    if num_pairs:
        average /= float(num_pairs)
    # diameter will be zero if graph is empty or largest connected component is of size 1
    feats['TEXT_di_{}'.format(graph_type)] = longest
    # calculate average shortest path (ASP)
    feats['TEXT_asp_{}'.format(graph_type)] = average


def get_graph_metrics(graph, graph_type, feats):
    """
    Computes features for the input graph. Features include: num_nodes: the number of nodes present in the graph,
    num_edges: number of edges in the graph, num_p_edges: number of parallel edges present in the graph,
    lcc: number of nodes in the largest connected component of the graph,
    lsc: number of nodes in the largest strongly connected component of the graph,
    atd: average degree of the nodes in the graph, l1: number of self-loops, l2: number of loops with two nodes,
    l3: number of triangles (an approximation to the number of loops with three nodes), and graph density.
    Graph connectivity measures(ASP and diameter) are also computed.
    :param graph: a directed multigraph
    :param graph_type: naive, lemma, or POS
    :param feats: dictionary to store feature values for the transcript
    """
    # calculate number of nodes
    num_nodes = len(graph)
    feats['TEXT_num_nodes_{}'.format(graph_type)] = num_nodes
    # calculate number of edges
    feats['TEXT_num_edges_{}'.format(graph_type)] = graph.number_of_edges()
    # get undirected graph
    u_graph = graph.to_undirected()
    get_connectivity_measures(graph, u_graph, graph_type, feats)
    num_p_edges, pe_l1_count = get_parallel_edges(graph, graph_type, feats)
    # calculate number of self-loops (L1)
    #l_one = len(list(graph.selfloop_edges()))
    l_one = len(list(nx.selfloop_edges(graph))) #networkx v2.5
    feats['TEXT_l1_{}'.format(graph_type)] = l_one
    #get_loops(graph, graph_type, feats)
    # calculate graph density (D)
    # This measure is defined for simple graphs. Therefore, we take E' = E - (L1 + PE).
    # i.e. duplicate edges in same direction only count once and self-loops are not counted
    e_prime = graph.number_of_edges() - (l_one + num_p_edges - pe_l1_count)
    if e_prime < 0:
        feats['TEXT_d_{}'.format(graph_type)] = float('nan')
    elif num_nodes:
        feats['TEXT_d_{}'.format(graph_type)] = e_prime / float(num_nodes * num_nodes)
    else:
        feats['TEXT_d_{}'.format(graph_type)] = float('nan')
    get_shortest_path_metrics(u_graph, graph_type, feats)


def get_word_count(segments):
    count = 0
    for segment in segments:
        count += len(segment)
    return count


def add_norm_feats(feats, word_count):
    """
    :param feats: Dictionary mapping feature name to value for transcript
    :param word_count: Transcript word count
    """
    for feat, value in list(feats.items()):
        feats["TEXT_{}_norm".format(feat)] = (float(value) / float(word_count)) if word_count else float('nan')


def extract_graph(df_c, feats):
    """
    :param segments: List of text segments. Each segment is a string.
    :return: feats: Dictionary mapping feature names to values
    """
    df_c = df_c.fillna(value={'transcriptions':' '})
    segments = [x.replace('<comma>', ',').replace('<filler>', ' ') for x in list(df_c['transcriptions'])]
    # if segments are all lowercase, try to infer true case (this helps POS detection)
    for idx, seg in enumerate(segments):
        if seg.islower():
            segments[idx] = truecase.get_true_case(seg)
    # break segments up into words
    #segments_mixed_case = [s.split(" ") for s in segments]
    segments_mixed_case = []
    for s in segments: 
        text = s.split(" ") 
        if '' in set(text):
            text[:] = [w for w in text if w != '']
        segments_mixed_case.append(text) 
    
    # also get lowercase version (used for naive graph)
    #segments_lower_case = [s.lower().split() for s in segments]
    segments_lower_case = []
    for s in segments: 
        text = s.lower().split() 
        if '' in set(text):
            text[:] = [w for w in text if w != '']
        segments_lower_case.append(text) 
    
    # build graphs
    naive_graph = create_naive_graph(segments_lower_case)
    lemma_graph = create_lemma_graph(segments_mixed_case)  # POS detection is used to help with lemmatization
    pos_graph = create_pos_graph(segments_mixed_case)
    # compute features for each graph
    get_graph_metrics(naive_graph, 'naive', feats)
    get_graph_metrics(lemma_graph, 'lemma', feats)
    get_graph_metrics(pos_graph, 'pos', feats)
    # add normalized versions of features
    word_count = get_word_count(segments)
    add_norm_feats(feats, word_count)
    return feats

