from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from typing import List, Dict, Tuple
import pytrec_eval
import pathlib, os
import logging
import csv

#### code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download trec-covid.zip dataset and unzip the dataset (change name of dataset for a different Beir data)
dataset = "trec-covid"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)


corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")


#### load in AskMe answers to be evaluated (change path of AskMe results TSV file) ####

askMeResults = {}

reader = csv.reader(open("PATH TO ASKME RESULTS FILE", encoding="utf-8"),
                    delimiter="\t", quoting=csv.QUOTE_MINIMAL)
next(reader)

for id, row in enumerate(reader):
    query_id, corpus_id, score = row[0], row[1], float(row[2])

    if query_id not in askMeResults:
        askMeResults[query_id] = {corpus_id: score}
    else:
        askMeResults[query_id][corpus_id] = score

#### evaluation function modified from Beir for AskMe ####
def evaluate(qrels: Dict[str, Dict[str, int]],
             results: Dict[str, Dict[str, float]],
             k_values: List[int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    for eval in [ndcg, _map, recall, precision]:
        logging.info("\n")
        for k in eval.keys():
            logging.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format([1,3,5,10,100,1000]))
ndcg, _map, recall, precision = evaluate(qrels, askMeResults, [1,3,5,10,100,1000])

