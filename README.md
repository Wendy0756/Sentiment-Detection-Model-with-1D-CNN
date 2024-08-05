<a id="readme-top"></a>

<br />
<div align="center">
  <h3 align="center">Emotion Detecting Model on Tweets</h3>

  <p align="center">
    Bernardo Cobos, Rebecca Bargiachi, William Lei, Xueying Tian
  </p>
</div>

## About The Project
The team builds a binary classification model with 1D Convolutional Neural Networks (CNN) architecture. The model classifies plain texts into positive and negative categories with a ~75% accuracy. 

### Data Source
tweets.csv file can be downloaded here: https://www.kaggle.com/datasets/kazanova/sentiment140

### Execution Environment
We strongly suggest using Google Colab to execute the notebook and leveraging a GPU environment.

### Libraries to install
**pandas, numpy, matplotlib, seaborn**

**sklearn:**

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

**Tensorflow:**

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout, Conv1D, MaxPooling1D

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping

**Gensim:**

from gensim.downloader as api

### Useful Articles
* [A Dummy's Guide to Word2Vec](https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673)
* [Emojis Aid Social Media Sentiment Analysis: Stop Cleaning Them Out!](https://towardsdatascience.com/emojis-aid-social-media-sentiment-analysis-stop-cleaning-them-out-bb32a1e5fc8e)
* [What is Emotion Detection? What are the methods used for Emotion Detection in Text Analytics?](https://textrics.medium.com/what-are-the-methods-used-for-emotion-detection-in-text-analytics-838d7ca7e435#:~:text=There%20are%20four%20different%20text,based%20method%2C%20and%20Hybrid%20methods.)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
