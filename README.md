BiDirectional LSTMs and Attention with Context for Detecting Wikipedia Toxic Comments
---------------------------------------------------------------------------------------
Wikipedia, the largest and most popular general reference work on the World Wide Web, is based on open collaboration. Open collaboration has its own difficulties since it enables insertion of contents that are toxic, due to identity hate, improper language or obscenetiy.

This project aims to build machine Learning models that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate by processing comments.

The dataset is multi-label, and highly imbalanced, and many times mislabeled. Initially, baseline models such as Naïve-Bayes, Logistic Regression, SVC were utilized, however these models did not perform well and failed to learn dependencies in explaining label sets “threat” and “identity hate”.

Following baseline models, various neural networks were trained after Keras text-preprocessing. At this time no other preprocessing methods were used, however, in later stages of the project, Spacy based text processing including lemmatization was made, with and without removing the stop words. Of all three instances, best performance was achieved with Spacy preprocessing without removing the stop words, followed by Keras text-preprocessing.

Due to extreme class imbalance, both class weights and synthetic oversampling methods were used when building single label models. Of all oversampling techniques, ADASYN performed the best, and thus all singular models were built with ADASYN. Over sampler was trained after padded sequences were created and was fit on model for each single label column that had high class imbalance. 

Neural networks evaluated for this project include feed forward dense, CNN, LSTM, GRU, BiDirectional LSTM and BiDirectional GRUs. Best results were achieved with dual BiDirectional LSTMs followed by an Attention with context layer. Despite lack of some important arguments, CudNNLSTM, a GPU-accelerated version for deep neural networks was used since it provided highly tuned implementations for standard routines, allowing models to run many times faster. 

The biggest problem ran into during this project was mislabeled instances. The model can only work effectively if the labels it is introduced are precise, and would produce erroneous results when fed with arbitrary labels. Unfortunately, during EDA analysis of this dataset, many comment texts were found to be mislabeled and it is impossible to assess the ratio or the impact on the model before reading all comments.

Due to the imbalance in the dataset, ROC AUC score would produce meaningless results, and thus was not preferred. Instead, the model performance was measured by precision recall auc.

The model produced good results with BiDirectional LSTMs and Attention layer, and showed significant improvement with introduction of oversampling. Various activations, number of hidden layers, dropout percentages, batch sizes were tried to achieve these results, however there could still be room for improvement on the hyperparameter tuning side.

For future work, planned area of improvement is using word embeddings to build a synonyms dictionary for data augmentation, which would be alleviate class imbalance.

Other are of planned future work includes using Bidirectional Encoder Representations from Transformers, or BERT, which will be handled as a separate project.


-----------------------------------------------------------------------------------------------------------------------------------

Models were not saved since Keras can't save custom layers, but weights were saved and can be found in "saved_model_weights" folder.

Text pre-processed csv files can be found in "preprocessed_data" folder. 7-zip will be required to unzip the files.

For wikipedia comment source files please visit:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


