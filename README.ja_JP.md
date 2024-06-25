# 強化学習によるゲーム戦略の最適化：スネークゲームのケーススタディ

<p align="center">
  <a href="README.md">English</a> •
  <a href="README.ja_JP.md">日本語 (Japanese)</a> •
  <a href="README.zh_CN.md">简体中文 (Simplified Chinese)</a> 
</p>

## 概要

本研究では、古典的なゲームである「スネークゲーム」をケーススタディとして、強化学習のゲーム戦略開発への応用に焦点を当てている。 研究の目標には、ゲームを効果的にプレイできるAIエージェントの開発、異なる強化学習アルゴリズムの有効性の比較、より広いAI領域におけるこれらの技術の可能性の探求が含まれる。 DQN、Double DQN、Dueling DQNの3つのアルゴリズムを用いて、いくつかの性能指標を用いたシミュレーション環境でAIエージェントを訓練した。 本研究の目的は、ゲーム戦略開発における強化学習の応用に関する理解を深め、より複雑な状況に対応できるフレームワークを提供し、より適応性の高いAIシステムの開発を推進することである。

## CONDAでランタイム環境を作成する

このファイルから `snake-ai-pytorch.yml` 環境を作成する：

```shell
conda env create -f snake-ai-pytorch.yml
```

新しい環境をアクティブにする: `conda activate snake-ai-pytorch`

新しい環境が正しくインストールされていることを確認する：

```shell
conda env list
```

`conda info --envs`を使うこともできる。

詳細については、以下を参照のこと：[CONDA User guide  > Managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## ANACONDAでランタイム環境を作成する【おすすめ】

ANACONDAダウンロードアドレス：[https://www.anaconda.com/download](https://www.anaconda.com/download)

`ANACONDA.NAVIGATOR`を起動し、`Environments`>`Import`で`snake-ai-pytorch.yml`をインポートする

![](assets_README.zh_CN/2024-06-25-11-41-49-image.png)

## ゲームを実行し、トレーニングプロセスを開始する

Pythonを使って`agent.py`を実行する：

```shell
python agent.py
```

ゲームを実行しているスクリーンショット：

![](assets_README.zh_CN/2024-06-25-11-48-31-image.png)

## トレーニング結果

トレーニング結果は `results` ディレクトリで見ることができる。

## 参考コード

スネークゲームの設計とDQNアルゴリズムの実装は、以下のプロジェクトのコードを参考とした：

[GitHub - patrickloeber/snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch)

Double DQNとDueling DQNアルゴリズムは、以下のプロジェクトのコードを参考に実装されている：

[GitHub - p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch: PyTorch implementations of deep reinforcement learning algorithms and environments](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
