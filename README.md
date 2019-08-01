# text-classifier

A Python3 software application for text classification with Word2vec and Gated Convolutional Networks (GCNN). 

## Introduction
The input text first undergoes data cleansing steps such as 
- Converting English letter to lower case 
- Translating simplified Chinese to traditional Chinese
- Segmenting of Chinese vocabularies, etc. 

Subsequently, the cleansed text is converted to word vectors via the pre-trained Word2vec model. The word vectors are then fed into the Neural Network model to compute the probabilities of the input text belonging to each pre-defined label. The label with the highest prediction probability is the final prediction. 

## Prerequisite
Below are the required Python dependencies for running the program:
```
bert-tensorflow==1.01
dbfread==2.0.7
feather-format==0.4.0
gensim==3.7.3
jieba==0.39
json==2.0.9
keras==2.2.4
numpy==1.16.3
opencc==0.2
opencc-python-reimplemented==0.1.4
pandas==0.24.2
sklearn==0.21.1
tensorflow-gpu==1.12.0
```
Install the dependencies by running `$ pip install -r requirements.txt`
**Note** : If the PC/server is not equipped with GPU card(s), install `tensorflow` instead of `tensorflow-gpu`.


## Implementation
The Python3 program can be executed on either Linux or Windows **command line** with 5 different modes, namely:

- `do_word2vec` : Train Word2vec vectors using a text corpus.
- `do_train` : Train Neural Network model for text classification.
- `do_eval` : Evaluate classification performance of the trained Neural Network model.
- `do_predict` : Prediction for a batch of new input data.
- `do_single_predict` : Single prediction on the command line.

There are about 50 arguments available to the main program of the model, `textclassifier.py`, some have default values while some do not. See **Appendix II** for more detailed description of the arguments.


### Code
Run the following codes on **Linux** command line to execute various modes of the application. 

> **do_word2vec**
```
$ python textclassifier.py --train_word2vec_file=file_path \
	--output_dir=file_path \
	--jieba_user_dict=file_path \
	--bert_vocab_file=file_path \
	--word2vec_sg=True \
	--word2vec_size=50 \
	--word2vec_window=10 \
	--word2vec_min_count=10 \
	--word2vec_iterations=100 \
	--word_tokenization=space or word_piece \
	--num_threads=12 \
	--do_word2vec=True
```
**Output Files**: 
The following documents will be saved at the file path specified for the argument `output_dir` :
- `EmbeddingVectors.feather` : A pandas data frame, each column is the trained word vector and the column name is the word token. 
- `Word2vec.model` : A Word2vec model object.
- `Word2vec.model.trainables.syn1neg.npy` : A Word2vec model object.
- `Word2vec.model.wv.vectors.npy` : A Word2vec model object.
- `WORD2VEC_FLAGS.json` : A `.json` file that stores the hyperparameters of the model.

***

> **do_train**
```
$ python textclassifier.py --model_name=model_name\
	--output_dir=file_path \
	--train_file=file_path \
	--valid_file=file_path \
	--class_file=file_path \
	--emb_matrix_file=file_path \
	--jieba_user_dict=file_path \
	--bert_vocab_file=file_path \
	--max_len=256 \
	--batch_size=128 \
	--num_filter_1=256 \
	--kernel_size_1=5 \
	--num_stride_1=1 \
	--num_filter_2=256 \
	--kernel_size_2=3 \
	--num_stride_2=1 \
	--hidden_dims_1=512 \
	--hidden_dims_2=256 \
	--hidden_dims_3=128 \
	--hidden_dims_4=64 \
	--epochs=100 \
	--learning_rate=0.001 \
	--es_patience=10 \
	--es_min_delta=0.0001 \
	--reduce_lr_patience=2 \
	--reduce_lr_factor=0.5 \
	--dropout_rate=0.5 \
	--loss_function=categorical_crossentropy or focal_loss \
	--word_tokenization=space or word_piece \
	--num_threads=12 \
	--train_with_weight=True \
	--do_train=True
```
**Output File** :
The following documents will be saved at the file path specified for the argument `output_dir` :
- `model_name.h5` : A `.h5` file that stores the trained parameters of the Neural Network model.
- `model_name_flags.json` : A `.json` file that stores the hyperparameters of the Neural Network model.
- `model_name_vocab.txt` : A `.txt` file that stores the corresponding vocabularies in the embedding layer of the Neural Network model.
- `invalid_label_train.feather` : A `.feather` file that contains data records in the training file which the labels are invalid (not present in the `class_file`).
- `invalid_label_valid.feather` : A `.feather` file that contains data records in the validation file which the labels are invalid (not present in the `class_file`).
- `uninformative_train.feather` : A `.feather` file that contains data records in the training file that are considered uninformative. A uninformative text input satisfy either one of the following conditions:
(i)		All word tokens are trimmed away during text cleansing process.
(ii)	All word tokens are not in the vocabulary list of the Word2vec model.
- `uninformative_valid.feather` : A `.feather` file that contains data records in the validation file that are considered uninformative.

***

> **do_eval**
```
$ python textclassifier.py \
	--keras_model_file=file_path \
	--keras_model_flags=file_path \
	--emb_layer_vocab=file_path \
	--output_dir=file_path \
	--eval_file=file_path \
	--class_file=file_path \
	--jieba_user_dict=file_path \
	--bert_vocab_file=file_path \
	--num_threads=12 \
	--do_eval=True
```

**Output Files** :
The following documents will be saved at the file path specified for the argument `output_dir` :
-  `invalid_label_eval.feather` : A `.feather` file containing data records in the evaluation file which the labels are invalid (not present in the `class_file`).
- `uninformative_eval.feather` : A `.feather` file containing data records in the evaluation file that are considered uninformative.
-  `evaluation.feather` : A `.feather` file containing the prediction and the prediction probabilities.
- `eval_metrics.csv` : A `.csv` file containing the performance statistics of the model over the evaluation dataset. Overall Classification rate and Recall/Precision/F-1 score at each group are available.

***

> **do_predict**
```
$ python textclassifier.py \
	--keras_model_file=file_path \
	--keras_model_flags=file_path \
	--emb_layer_vocab=file_path \
	--output_dir=file_path \
	--predict_file=file_path \
	--class_file=file_path \
	--jieba_user_dict=file_path \
	--bert_vocab_file=file_path \
	--num_threads=12 \
	--do_predict=True
```
**Output Files** :
The following documents will be saved at the file path specified for the argument `output_dir` :
- `uninformative_predict.feather` : A `.feather` file containing data records in the evaluation file that are considered uninformative.
- `prediction.feather` : A `.feather` file containing the label prediction and prediction probabilities. 

***

> **do_single_predict**
```
$ python textclassifier.py \
	--keras_model_file=file_path \
	--keras_model_flags=file_path \
	--emb_layer_vocab=file_path \
	--class_file=file_path \
	--jieba_user_dict=file_path \
	--bert_vocab_file=file_path \
	--num_threads=12 \
	--do_single_predict=True
```

**Note**:
-	If the PC/server is equipped with GPU, by default the program will run on GPU. 
-	To force the program to run on **CPU**, put the argument `CUDA_VISIBLE_DEVICES=-1` before the `python` keyword on command line.
-	A space character followed by a slash character, i.e. **" \\"** is added to the end of each command line to start a new line. Extra characters after **" \\"** in the same line will lead to errors.
-	 The codes to be entered in Windows command line is almost the same as Linux, except that **" ^"** instead of **" \\"** is used for starting a new line.


## Appendix II :  Arguments in the Python3 main program `textclassifier.py`
The table below is the available arguments of the model:

| ARGUMENT            | DESCRIPTION                                                                                                                                                                                                                                                                        | DEFAULT VALUE |
|:--------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------|
| model_name          | The name of the Neural Network Model.                                                                                                                                                                                                                                               | None          |
| train_file          | The path of `.feather/.csv/.xlsx/.dbf` file that contains the training data.                                                                                                                                                                                                       | None          |
| valid_file          | The path of `.feather/.csv/.xlsx/.dbf` file that contains the validation data.                                                                                                                                                                                                     | None          |
| eval_file           | The path of `.feather/.csv/.xlsx/.dbf` file that contains the data for model performance evaluation.                                                                                                                                                                               | None          |
| predict_file        | The path of `.feather/.csv/.xlsx/.dbf` file that contains the data for prediction.                                                                                                                                                                                                 | None          |
| train_word2vec_file | The path for `.feather/.csv/.xlsx/.dbf` file that contains the training corpus for training Word2vec model.                                                                                                                                                                                 | None          |
| output_dir          | The directory that stores the output from each mode. It is created automatically if not exist.                                                                                 | None          |
| class_file          | The path of `.csv` file that contains the metadata of the classes of the classification task.                                                                                                                                                                                                      | None          |
| emb_matrix_file     | The path of `.feather` file that stores the word embedding matrix.                                                                                                                                                                                                               | None          |
| jieba_user_dict     | The path of additional vocabulary file (`.txt`) for customized Chinese segmentation                                                                                                                                                                                                | None          |
| keras_model_file    | The path of `.h5` file that stores the trained KERAS Neural Network model.                                                                                                                                                                                                       | None          |
| keras_model_flags   | The path of `.json` file that contains the hyperparameters of the trained KERAS Neural Network model.                                                                                                                                                                              | None          |
| keras_model_vocab   | The path of `.txt` file that contains the vocabularies in the embedding layer of the trained Neural Network model.                                                                                                                                                                 | None          |
| bert_vocab_file     | The BERT vocabulary file (`.txt`) for Word Piece tokenization, it must exist in `do_train` mode if `word_segmentation=word_piece`. It must also exists in `do_eval`, `do_predict` and `do_single_predict` mode if the pretrained Neural Network model to be called uses Word Piece tokenization. | None          |
| num_threads         | The number of CPU cores for multiprocessing.                                                                                                                                                                                                                                       | 12            |
| do_word2vec         | Whether to train Word2vec vectors.                                                                                                                                                                                                                                              | False         |
| do_train            | Whether to train Neural Network model.                                                                                                                                                                                                                                                           | False         |
| do_eval             | Whether to run evaluation using the evaluation dataset.                                                                                                                                                                                                                               | False         |
| do_predict          | Whether to run prediction on the input dataset.                                                                                                                                                                                                                                    | False         |
| do_single_predict   | Whether to run prediction on single entry on command line.                                                                                                                                                                                                                         | False         |
| train_with_weights  | Whether to apply weights to loss function during Neural Network training. If the values is **True**. There must be a numerical column named **WEIGHT** in the `class_file`                                                                                                           | True          |
| word_tokenization   | Which word tokenization method used in text cleansing, must be either `word_piece` or `space`.                                                                                                                                                                                     | None          |
| word2vec_sg         | True for training Skip-gram model and False for training CBOW.                                                                                                                                                                                                                     | None          |
| word2vec_size       | Dimension of Word2vec vectors.                                                                                                                                                                                                                                                     | None          |
| word2vec_window     | Maximum distance between the current and predicted word within a sentence.                                                                                                                                                                                                         | None          |
| word2vec_min_count  | Ignores all words with total frequency lower than this in Word2vec training.                                                                                                                                                                                                       | None          |
| word2vec_iterations | Number of iterations (epochs) over the corpus in Word2vec training.                                                                                                                                                                                                                | None          |
| max_len             | Maximum length of token sequence. Tokens beyond the threshold are trimmed.                                                                                                                                                                                                         | 256           |
| batch_size          | Training batch size.                                                                                                                                                                                                                                                               | 128           |
| num_filter_1        | The number of Convolutional filters in branch 1 of GCNN.                                                                                                                                                                                                                           | 256           |
| kernel_size_1       | The length of the 1D convolution window in branch 1.                                                                                                                                                                                                                               | 5             |
| num_stride_1        | The stride length of the convolution in branch 1.                                                                                                                                                                                                                                  | 1             |
| num_filter_2        | The number of Convolutional filters in branch 2 of GCNN.                                                                                                                                                                                                                           | 256           |
| kernel_size_2       | The length of the 1D convolution window in branch 2.                                                                                                                                                                                                                               | 3             |
| num_stride_2        | The stride length of the convolution in branch 2.                                                                                                                                                                                                                                  | 1             |
| hidden_dims_1       | The number of neurons in the 1st feed-forward Neural Network layer.                                                                                                                                                                                                                | 512           |
| hidden_dims_2       | The number of neurons in the 2nd feed-forward Neural Network layer.                                                                                                                                                                                                                | 256           |
| hidden_dims_3       | The number of neurons in the 3rd feed-forward Neural Network layer.                                                                                                                                                                                                                | 128           |
| hidden_dims_4       | The number of neurons in the 4th feed-forward Neural Network layer.                                                                                                                                                                                                                | 64            |
| epochs              | The number of training epochs to perform.                                                                                                                                                                                                                                          | 100           |
| learning_rate       | The initial learning rate for Adam.                                                                                                                                                                                                                                                | 0.001         |
| es_patience         | The number of epochs with no improvement after which training will be stopped.                                                                                                                                                                                                     | 10            |
| es_min_delta        | The threshold for measuring the new optimum, to only focus on significant changes.                                                                                                                                                                                                 | 1e-04         |
| reduce_lr_patience  | If no improvement is seen for such number of epochs, the learning rate is reduced.                                                                                                                                                                                                 | 2             |
| reduce_lr_factor    | Factor by which the learning rate will be reduced. new_lr = lr * factor                                                                                                                                                                                                            | 0.5           |
| dropout_rate        | The fraction of input units setting at 0 at each update during training time, which helps prevent over-fitting.                                                                                                                                                                    | 0.5           |
| loss_function       | Loss function for training Neural Network, can only be either `categorical_crossentropy` or `focal_loss`.                                                                                                                                                                          | None          |

## Input Files
|FILE                  |DESCRIPTION                                                                                                                                                                                                                                                                                                      |
|:---------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`train_word2vec_file` |A `.feather/.csv/.xlsx/.dbf` file. There is a column named **TEXT** which corresponds to the text corpus for training Word2vec model.                                                                                                                                                                                               |
|`train_file`          |A `.feather/.csv/.xlsx/.dbf` file. There is a column named **TEXT** which corresponds to the text for classification. Another column called **LABEL** is required, which corresponds to the label of the text. It is used for training the Neural Network model.                                                                                                                           |
|`valid_file`          |A `.feather/.csv/.xlsx/.dbf` file. There is a column named **TEXT** which corresponds to the text for classification. Another column called **LABEL** is required, which corresponds to the label of the text. It is used for validation during the training of Neural Network model.                                                                                                                           |
|`eval_file`           |A `.feather/.csv/.xlsx/.dbf` file with the **TEXT** column corresponding to the text for classification and the **LABEL** column corresponding to the label of the text. It is used for evaluating the prediction performance of the trained Neural Network model.                                                                                                                            |
|`predict_file`           |A `.feather/.csv/.xlsx/.dbf` file with the **TEXT** column corresponding to the text for classification.                                                                                                                                                                                                   |
|`keras_model_file`       |A `.h5` file that stores the trained Keras Neural Network model.                                                                                                                                                                                                   |
|`keras_model_flags`      |A `.json` file that stores the hyperparameters of the trained Keras Neural Network model.                                                                                                                                                                                                  |
|`keras_model_vocab`      |A `.txt` file that stores the corresponding vocabularies of the embedding layer of the Neural Network model. Each vocabulary is placed on a new line in the file.                                                                                                                                                                                               |
|`class_file`          |A `.csv` file with 3 columns, namely **LABEL**, **DESCRIPTION** and **WEIGHT**. **LABEL** is the set of labels for the classification task. **DESCRIPTION** is the description of the label. **WEIGHT** is the pre-determined (upon the researcher's choice) weight assigned to each label in training the Neural Network model. |
|`jieba_user_dict`     |A `.txt` file that contains custom Chinese vocabularies for customized Chinese text segmentation in text cleansing. Each vocabulary is placed on a new line in the file.                                                                                                                                                                                                                    |
|`bert_vocab_file`     |a `.txt` file for performing Word Piece tokeniztion during text cleansing. Various vocabulary files are available at  [google-bert pretrained models](https://github.com/google-research/bert#pre-trained-models).                                                                                                                                            |
