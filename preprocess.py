#%%
import numpy as np 
import pandas as pd
from statistics import mode
validation = pd.read_csv('validation/dialogues_validation.txt', delimiter='\n', header=None)
test = pd.read_csv('test/dialogues_test.txt', delimiter='\n', header=None)
train = pd.read_csv('train/dialogues_train.txt', delimiter='\n', header=None)
val_labels = pd.read_csv('validation/dialogues_emotion_validation.txt', delimiter='\n', header=None)
test_labels = pd.read_csv('test/dialogues_emotion_test.txt', delimiter='\n', header=None)
train_labels = pd.read_csv('train/dialogues_emotion_train.txt', delimiter='\n', header=None)


print(train.shape)
print(validation.shape)
print(test.shape)
print(train_labels.shape == train.shape)
print(val_labels.shape == validation.shape)
print(test_labels.shape == test.shape)
# %%
df_train = pd.DataFrame({'text':train[0].values,'emotions':train_labels[0].values})
df_train.head()
df_train['selected'] = df_train['emotions'].apply(lambda x: max(x.split(), key=x.split().count))
# df_train['mode_emotion'] = df_train['emotions'].apply(lambda x: mode(x.split())) same as the above one
# %%
def split_sent(df):
    sent = []
    for text in df['text']:
        sent.extend(text.split('__eou__'))
    return sent
def label_split(df):
    labels = []
    for emotion in df['emotions']:
        labels.extend(emotion.split())
    return labels
# %%
sent = split_sent(df_train)
labels = label_split(df_train)
clear = [s for s in sent if len(s)>3]
sent_df = pd.DataFrame({'sent': clear, 'labels': labels})
sent_df.groupby('labels').size().plot(kind='bar')
print('text_level', df_train['selected'].value_counts())
print('sentence level', sent_df['labels'].value_counts())
# %%
sent_df.groupby('labels').size().plot(kind='bar')
#%%
df_train.groupby('selected').size().plot(kind='bar')
# %%
