# DOWNLOAD DATASETS
cd datasets/

## Simple tabular housing dataset
if [ ! -d "bostonhousing" ]; then
    echo -e "Downloading Boston Housing dataset"
    kaggle datasets download -d schirmerchad/bostonhoustingmlnd
    unzip bostonhoustingmlnd.zip -d bostonhousing
    rm bostonhoustingmlnd.zip
fi

## basics image -- mnist
if [ ! -d "mnist" ]; then
    echo -e "Downloading MNIST (csv) dataset (competition)"
    kaggle competitions download -c digit-recognizer
    unzip digit-recognizer.zip -d mnist
    rm digit-recognizer.zip
fi

## emotional speech dataset
if [ ! -d "emotionalspeech" ]; then
    echo -e "Downloading Emotional Speech dataset"
    kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
    unzip ravdess-emotional-speech-audio.zip -d emotionalspeech
    rm ravdess-emotional-speech-audio.zip
fi

## stretch -- $160,000 Kaggle Competition; natural language processing
if [ ! -d "feedbackprize" ]; then
    echo -e "Downloading Feedback Prize dataset (competition)"
    kaggle competitions download -c feedback-prize-2021
    unzip feedback-prize-2021.zip -d feedbackprize
    rm feedback-prize-2021.zip
fi

## mnist sign language
if [ ! -d "mnist_sign_language" ]; then
    echo -e "Downloading MNIST Sign Language dataset"
    kaggle datasets download -d datamunge/sign-language-mnist
    unzip sign-language-mnist.zip -d mnist_sign_language
    rm sign-language-mnist.zip
fi
