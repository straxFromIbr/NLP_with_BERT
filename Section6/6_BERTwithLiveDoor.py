#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
ON_COLAB = "COLAB_GPU" in os.environ


# In[2]:


from datetime import date
modelpath = f"./model_{date.today().strftime('%y-%m-%d')}"
if ON_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    modelpath = '/content/drive/MyDrive/ColabFolder/NLPwithBERT/Section6/'+modelpath


# ## ライブラリインストール

# In[3]:


if ON_COLAB:
    get_ipython().system(' pip install -U pip 2>&1 >/dev/null')
    get_ipython().system(' pip install         transformers==4.5.0         fugashi==1.1.0         ipadic==1.0.0          pytorch-lightning==1.2.7 2>&1 >/dev/null ')


# ## データセットのダウンロードと解凍

# In[4]:


if not os.path.exists('ldcc-20140209.tar.gz'):
    get_ipython().system(" wget 'https://rondhuit.com/download/ldcc-20140209.tar.gz'  >/dev/null 2>&1")
    get_ipython().system(" tar -zxf 'ldcc-20140209.tar.gz'  >/dev/null 2>&1")


# In[5]:


get_ipython().system(" head -n7 './text/it-life-hack/it-life-hack-6342280.txt'")


# In[6]:


import random
import glob
from tqdm import tqdm

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"


# In[7]:


# ラベルリスト
category_list = [
    "dokujo-tsushin",
    "it-life-hack",
    "kaden-channel",
    "livedoor-homme",
    "movie-enter",
    "peachy",
    "smax",
    "sports-watch",
    "topic-news",
]
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)


# In[8]:


# データ整形
max_length = 128
dataset_for_loader = []
for label, category in enumerate(tqdm(category_list)):
    for file in glob.glob(f"./text/{category}/{category}*"):
        lines = open(file).read().splitlines()
        text = "\n".join(lines[3:])
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        encoding["labels"] = label
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        dataset_for_loader.append(encoding)

print(dataset_for_loader[0])


# In[9]:


random.shuffle(dataset_for_loader)
n = len(dataset_for_loader)
n_train = int(0.6 * n)
n_val = int(0.2 * n)
dataset_train = dataset_for_loader[:n_train]
dataset_val = dataset_for_loader[n_train : n_train + n_val]
dataset_test = dataset_for_loader[n_train + n_val :]

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=256)
dataloader_test = DataLoader(dataset_test, batch_size=256)


# ## PyTorch Lightningで文章分類モデルを構築する

# In[10]:


class BertForSequenceClassification_pl(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr) -> None:
        """
        model_name: モデルの名前
        """
        super().__init__()
        # `__init__`の引数を保存する！便利！
        self.save_hyperparameters()

        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def training_step(self, batch, batch_idx):
        """
        # 各学習ステップで呼ばれる関数
            - 損失を記録し、返す
        """
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        # こちらは検証ステップで呼ばれる関数
            - 損失を記録し、返す
        """
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """
        # こちらはテストステップで呼ばれる関数
            - ラベルの正解率を求めて記録する
        """
        labels = batch.pop("labels")
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = (labels_predicted == labels).sum().item()
        # 正解率を求める
        accurancy = num_correct / labels.size(0)
        self.log("accurancy", accurancy)
    
    def configure_optimizers(self):
        """
        オプティマイザを返す。
        オプティマイザにはAdamを使用しモデルのパラメータと学習率を渡す
        """
        return torch.optim.Adam(self.parameters() ,lr=self.hparams.lr)


# ## CheckpointのCallbackとTrainerの作成
# - kerasとノリが似てる

# In[11]:


checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True,
    dirpath=modelpath,
)

if ON_COLAB:
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
        callbacks=[checkpoint],
    )


# In[12]:


# モデル作成
model = BertForSequenceClassification_pl(MODEL_NAME, num_labels=9, lr=1e-5)


# In[13]:


# 訓練(ファインチューニング)する
hist = trainer.fit(model, dataloader_train, dataloader_val)


# In[14]:


best_model_path = checkpoint.best_model_path
print('最良モデルのチェックポイント', best_model_path)
print('裁量モデルでの検証データに対する損失', checkpoint.best_model_score)


# In[15]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir ./lightning_logs')


# In[15]:


get_ipython().system(' mv ./lightning_log /content/drive/MyDrive/ColabFolder/NLPwithBERT/Section6')

