from openicl import PromptTemplate
from openicl import DatasetReader
from openicl import RandomRetriever, BM25Retriever, ConERetriever, TopkRetriever, PPLInferencer, AccEvaluator
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator


def main(template, train_path, test_path, model_path, sentence_model_path, input_columns_name, output_columns_name, ice_num, candidate_num, select_time, batch_size, seed, output_json_filepath):
    # load dataset
    combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})

    train_dataset = combined_dataset["train"]
    test_dataset = combined_dataset["test"]

    # Print some information about the datasets
    print(train_dataset)
    print(test_dataset)

    accelerator = Accelerator()

    # Construct the DatasetReader
    data = DatasetReader(combined_dataset, input_columns=input_columns_name, output_column=output_columns_name)

    # different retrival stratigies
    topk_retriever = TopkRetriever(data, ice_num=ice_num, sentence_transformers_model_name=sentence_model_path, tokenizer_name=sentence_model_path, batch_size=batch_size, accelerator=accelerator)
    cone_retriever = ConERetriever(data, ice_num=ice_num, candidate_num=candidate_num, sentence_transformers_model_name=sentence_model_path, tokenizer_name=sentence_model_path, model_tokenizer_name=model_path, ce_model_name=model_path, ice_template=template, select_time=candidate_num, seed=seed, batch_size=batch_size, test_split='test', accelerator=accelerator)
    
    print("Start inference....")

    inferencer = PPLInferencer(model_name=model_path, tokenizer=model_path,output_json_filepath=output_json_filepath, batch_size=batch_size, accelerator=accelerator)

    # inference
    topk_predictions = inferencer.inference(topk_retriever, ice_template=template, output_json_filename=f'topk_seed_{seed}_{ice_num}_shot')
    cone_predictions = inferencer.inference(cone_retriever, ice_template=template, output_json_filename=f'topk_cone_seed_{seed}_{candidate_num}_{ice_num}_shot')


if __name__ == '__main__':
    subj_tp_dict = {
        0: "</E>Input: </text> Type: objective",
        1: "</E>Input: </text> Type: subjective"
    }
    subj_template = PromptTemplate(subj_tp_dict, {'text': '</text>'}, ice_token='</E>')

    sst2_tp_dict = {
        0: "</E>Review: </text> Sentiment: negative",
        1: "</E>Review: </text> Sentiment: positive"
    }
    sst2_template = PromptTemplate(sst2_tp_dict, {'text': '</text>'}, ice_token='</E>')

    sst5_tp_dict = {
        0: "</E>Review: </text> Sentiment: terrible",
        1: "</E>Review: </text> Sentiment: bad",
        2: "</E>Review: </text> Sentiment: okay",
        3: "</E>Review: </text> Sentiment: good",
        4: "</E>Review: </text> Sentiment: great",
    }
    sst5_template = PromptTemplate(sst5_tp_dict, {'text': '</text>'}, ice_token='</E>')

    cr_tp_dict = {
        0: "</E>Review: </text> Sentiment: negative",
        1: "</E>Review: </text> Sentiment: positive"
    }
    cr_template = PromptTemplate(cr_tp_dict, {'text': '</text>'}, ice_token='</E>')

    ag_news_tp_dict = {
        0: "</E>Input: </text> Type: world",
        1: "</E>Input: </text> Type: sports",
        2: "</E>Input: </text> Type: business",
        3: "</E>Input: </text> Type: technology",
    }
    ag_news_template = PromptTemplate(ag_news_tp_dict, {'text': '</text>'}, ice_token='</E>')

    mnli_tp_dict = {
        0: "</E></text1> Can we know </text>? Yes.",
        1: "</E></text1> Can we know </text>? Maybe.",
        2: "</E></text1> Can we know </text>? No."
        }
    mnli_template = PromptTemplate(mnli_tp_dict, {'text1': '</text1>', 'text2': '</text>'}, ice_token='</E>')

    qnli_tp_dict = {
        0: "</E></text1> Can we know </text>? Yes.",
        1: "</E></text1> Can we know </text>? No."
        }
    qnli_template = PromptTemplate(qnli_tp_dict, {'text1': '</text1>', 'text2': '</text>'}, ice_token='</E>')

    templates = {'sst2': sst2_template,
            'subj': subj_template,
            "sst5": sst5_template,
            'cr': cr_template,
            "ag_news": ag_news_template,
            "mnli": mnli_template,
            "qnli": qnli_template
            }

    input_columns={'sst2': ["text"],
            'subj': ['text'],
            "sst5": ["text"],
            "cr": ["text"],
            "ag_news": ["text"],
            'mnli': ['text1', 'text2'],
            "qnli": ["text1", "text2"]
            }

    output_columns={'sst2': 'label',
             'subj': 'label',
             "sst5": 'label',
             'cr': 'label',
             "ag_news": 'label',
             'mnli': 'label',
             "qnli": 'label'
            }

    test_split={
            'sst2': 'test',
            "subj": 'test',
            "sst5": 'test',
            "cr": 'test',
            "ag_news": 'test',
            'mnli': 'validation', # cannot get gold labels for the test split
            "qnli": 'validation',
    }

    task_names = ['sst2', 'subj']
    model_names = ['Llama-2-7b-hf']
    seeds = [1, 43, 666]

    # set the model and dataset path
    model_dir = ''
    sentence_transformer_path = ''
    data_dir = ''

    for model_name in model_names:
        model_path = model_dir + model_name
        sentence_model_path = sentence_transformer_path

        for seed in seeds:
            for task_name in task_names:
                train_path = data_dir + task_name + '/train.jsonl'
                test_name = test_split[task_name]
                test_path = data_dir + task_name + '/' + test_name + '.jsonl'

                ice_num = 8
                output_json_filepath = './results/' + model_name + '/' + task_name

                import os
                os.makedirs(output_json_filepath, exist_ok=True)

                batch_size = 10

                candidate_num = 30
                select_time = 10

                main(templates[task_name], train_path, test_path, model_path, sentence_model_path, input_columns[task_name], output_columns[task_name], ice_num, candidate_num, select_time, batch_size, seed, output_json_filepath)
