
# Introduce：  
This is a Uyghur language translator that supports speech-to-text in Uyghur language, machine translation to Uyghur language text, and text-to-speech in Uyghur language using AI. It also supports conversion between Latin Uyghur, Arabic Uyghur, and Chinese.

# Usage
### Installation
- python >= 3.8  
- torch  >= 1.13  
```
cd fairseq
pip install --editable ./ 
pip install tensorboardX

cd ..
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

```
### download nllb
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizerFast
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer_uig = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="uig_Arab")
model.save_pretrained("your path")
tokenizer_uig.save_pretrained("your path")