
# Introduce：  
This is a Uyghur language translator that supports speech-to-text in Uyghur language, and text-to-speech in Uyghur language. And Chinese to Uyghur and Uyghur to Chinese. Notice that we directly use the model from MMS (Meta) and NLLB (Meta).   
You can find everything from:  
https://huggingface.co/facebook/mms-1b-all  
https://github.com/facebookresearch/fairseq/tree/main/examples/mms#asr  
https://huggingface.co/facebook/nllb-200-distilled-600M  


# Usage
### Installation
- python >= 3.8  
- torch  >= 1.13  
```
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable ./ 
pip install tensorboardX

cd ..
git clone https://github.com/jaywalnut310/vits.git
cd vits
pip install Cython==0.29.21
pip install librosa
pip install phonemizer==2.2.1
pip install scipy
pip install numpy
pip install matplotlib
pip install Unidecode==1.1.1

cd monotonic_align/
mkdir monotonic_align
python3 setup.py build_ext --inplace
```

### Preparation
- checkpoint download
```
wget https://dl.fbaipublicfiles.com/mms/tts/uig-script_arabic.tar.gz
tar -zxvf uig-script_arabic.tar.gz
wget https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt'
```
### Example
ASR_TTS_Run.ipynb
