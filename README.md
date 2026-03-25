#  Classification (40 points)
Although often defined as a binary categorization problem, sentiment analysis is actually a complex task, or
suitcase research problem, as it requires tackling many other subtasks. Choose at least two subtasks to perform
information extraction on your crawled data. Unless you are sure that your data does not contain any neutral
content, you should always cover at least subjectivity detection and polarity detection. Namely, you should first
categorize your data as neutral versus opinionated and then classify the resulting opinionated data as positive
versus negative. Different classification approaches can be applied, including:
```
• knowledge based, e.g., SenticNet
• rule based, e.g., linguistic patterns
• machine learning based, e.g., deep neural networks
• hybrid (a combination of any of the above)
```
You can tap into any resource or toolkit you like, as long as you motivate your choices and you are able to
critically analyze obtained results. Some possible choices include:
```
• Weka: https://cs.waikato.ac.nz/ml/weka
• Hadoop: https://hadoop.apache.org
• Pylearn2: https://pylearn2.readthedocs.io/en/latest
• SciKit: https://scikit-learn.org
• NLTK: https://nltk.org
• Theano: https://github.com/Theano
• Keras: https://github.com/fchollet/keras
• Tensorflow: https://github.com/tensorflow/tensorflow
• PyTorch: https://pytorch.org
• Huggingface: https://huggingface.co/
• AllenNLP https://github.com/allenai/allennlp
```
# Question 4: Perform the following tasks:
• Motivate the choice of your classification approach in relation with the state of the art
• Discuss whether you had to preprocess data (e.g., microtext normalization) and why
• Build an evaluation dataset by manually labeling at least 1,000 records with an inter-annotator agreement
of at least 80% (it is recommended to have 3 annotators, but 2 is also OK)
• Provide evaluation metrics such as precision, recall, and F-measure on such dataset
• Perform a random accuracy test on the rest of the data and discuss results
• Discuss performance metrics, e.g., records classified per second, and scalability of the system

# Question 5: Explore some innovations for enhancing classification. 
If you introduce more than one, perform an ablation study to show the contribution of each innovation. For example, if you perform word sense disambiguation (WSD) and named entity recognition (NER) to enhance sentiment analysis, show the increase inaccuracy when adding only WSD, the increase in accuracy when adding only NER, and the increase in accuracy when adding both WSD and NER to your system. Explain why they are important to solve specific problems, illustrated with examples. Possible innovations include (but are not limited to) the following:
• Enhanced classification (add another sentiment analysis subtask, e.g., sarcasm detection)
• Fine-grained classification (e.g., perform aspect-based sentiment analysis)
• Hybrid classification (e.g., apply both symbolic and subsymbolic AI)
• Cognitive classification (e.g., use brain-inspired algorithms)
• Multitask classification (e.g., perform two sentiment analysis tasks jointly)
• Ensemble classification (e.g., use stacked ensemble)