#!/bin/bash

# DOWNLOAD DATASETS
cd datasets/

## Simple tabular housing dataset
if [ ! -d "bostonhousing" ]; then
    kaggle datasets download -d schirmerchad/bostonhoustingmlnd
    unzip bostonhoustingmlnd.zip -d bostonhousing
    rm bostonhoustingmlnd.zip
fi

## basics image -- mnist
if [ ! -d "mnist" ]; then
    kaggle competitions download -c digit-recognizer
    unzip digit-recognizer.zip -d mnist
    rm digit-recognizer.zip
fi

## stretch -- $160,000 Kaggle Competition; natural language processing
if [ ! -d "feedbackprize" ]; then
    kaggle competitions download -c feedback-prize-2021
    unzip feedback-prize-2021.zip -d feedbackprize
    rm feedback-prize-2021.zip
fi
