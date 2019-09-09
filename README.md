
# Pytorch-Transformers-Classification


This repository is based on the [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) library by HuggingFace. 

### Setup

1. Install Anaconda or Miniconda Package Manager from [here](https://www.anaconda.com/distribution/)
2. Create a new virtual environment and install packages.  
`conda create -n transformers python pandas tqdm jupyter`  
`conda activate transformers`  
If using cuda:  
  `conda install pytorch cudatoolkit=10.0 -c pytorch`  
else:  
  `conda install pytorch cpuonly -c pytorch`  
`conda install -c anaconda scipy`  
`conda install -c anaconda scikit-learn`  
`pip install pytorch-transformers` 

### Current Pretrained Models

The table below shows the currently available model types and their models. You can use any of these by setting the `model_type` and `model_name` in the `config` dictionary. For more information about pretrained models, see [HuggingFace docs](https://huggingface.co/pytorch-transformers/pretrained_models.html).

| Architecture        | Model Type           | Model Name  | Details  |
| :------------- |:----------| :-------------| :-----------------------------|
| BERT      | bert | bert-base-uncased | 12-layer, 768-hidden, 12-heads, 110M parameters.<br>Trained on lower-cased English text. |
| BERT      | bert | bert-large-uncased | 24-layer, 1024-hidden, 16-heads, 340M parameters.<br>Trained on lower-cased English text. |
| BERT      | bert | bert-base-cased | 12-layer, 768-hidden, 12-heads, 110M parameters.<br>Trained on cased English text. |
| BERT      | bert | bert-large-cased | 24-layer, 1024-hidden, 16-heads, 340M parameters.<br>Trained on cased English text. |
| BERT      | bert | bert-base-multilingual-uncased | (Original, not recommended) 12-layer, 768-hidden, 12-heads, 110M parameters. <br>Trained on lower-cased text in the top 102 languages with the largest Wikipedias |
| BERT      | bert | bert-base-multilingual-cased | (New, recommended) 12-layer, 768-hidden, 12-heads, 110M parameters.<br>Trained on cased text in the top 104 languages with the largest Wikipedias |
| BERT      | bert | bert-base-chinese | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>Trained on cased Chinese Simplified and Traditional text. |
| BERT      | bert | bert-base-german-cased | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>Trained on cased German text by Deepset.ai |
| BERT      | bert | bert-large-uncased-whole-word-masking | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>Trained on lower-cased English text using Whole-Word-Masking |
| BERT      | bert | bert-large-cased-whole-word-masking | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>Trained on cased English text using Whole-Word-Masking |
| BERT      | bert | bert-large-uncased-whole-word-masking-finetuned-squad | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>The bert-large-uncased-whole-word-masking model fine-tuned on SQuAD |
| BERT      | bert | bert-large-cased-whole-word-masking-finetuned-squad | 24-layer, 1024-hidden, 16-heads, 340M parameters <br>The bert-large-cased-whole-word-masking model fine-tuned on SQuAD |
| BERT      | bert | bert-base-cased-finetuned-mrpc | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>The bert-base-cased model fine-tuned on MRPC |
| XLNet      | xlnet | xlnet-base-cased | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>XLNet English model |
| XLNet      | xlnet | xlnet-large-cased | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>XLNet Large English model |
| XLM      | xlm | xlm-mlm-en-2048 | 12-layer, 2048-hidden, 16-heads <br>XLM English model |
| XLM      | xlm | xlm-mlm-ende-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-German Multi-language model |
| XLM      | xlm | xlm-mlm-enfr-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-French Multi-language model |
| XLM      | xlm | xlm-mlm-enro-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-Romanian Multi-language model |
| XLM      | xlm | xlm-mlm-xnli15-1024 | 12-layer, 1024-hidden, 8-heads <br>XLM Model pre-trained with MLM on the 15 XNLI languages |
| XLM      | xlm | xlm-mlm-tlm-xnli15-1024 | 12-layer, 1024-hidden, 8-heads <br>XLM Model pre-trained with MLM + TLM on the 15 XNLI languages |
| XLM      | xlm | xlm-clm-enfr-1024 | 12-layer, 1024-hidden, 8-heads <br>XLM English model trained with CLM (Causal Language Modeling) |
| XLM      | xlm | xlm-clm-ende-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-German Multi-language model trained with CLM (Causal Language Modeling) |
| RoBERTa      | roberta | roberta-base | 125M parameters <br>RoBERTa using the BERT-base architecture |
| RoBERTa      | roberta | roberta-large | 24-layer, 1024-hidden, 16-heads, 355M parameters <br>RoBERTa using the BERT-large architecture |
| RoBERTa      | roberta | roberta-large | 24-layer, 1024-hidden, 16-heads, 355M parameters <br>roberta-large fine-tuned on MNLI. |

### Custom Datasets

Please refer to [data-prep-compliance.ipynb](data-prep-compliance.ipynb) to convert the customer dataset to a Pytorch-Transformer ready format.

The data needs to be in `tsv` format, with four columns, and no header.

This is the required structure.

- `guid`: An ID for the row.
- `label`: The label for the row (should be an int).
- `alpha`: A column of the same letter for all rows. Not used in classification but still expected by the `DataProcessor`.
- `text`: The sentence or sequence of text.

### Train/Evaluate

% cd src

% python transformer-compliance.py ../config-bert.json

Depending on size of datasize, it may take a while.

### Evaluation Metrics

The evaluation process in the [transformer-compliance.py](transformer-compliance.py)  outputs the confusion matrix, and the Matthews correlation coefficient. The `get_eval_reports()` function takes the predictions and the ground truth labels as parameters, therefore you can add any custom metrics calculations to the function as required. Also the matrics will be saved under 'outputs' dir.
