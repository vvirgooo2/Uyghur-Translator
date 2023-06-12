from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from convertTool import converter

class Translator:
    def __init__(self, path, local_files_only, max_length):
        self.max_length = max_length
        # 初始化model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=local_files_only)
        # 初始化token 
        self.tokenizerArab2Hans = AutoTokenizer.from_pretrained(path, src_lang='uig_Arab')
        self.tokenizerHans2Arab = AutoTokenizer.from_pretrained(path, src_lang='zho_Hans')
        #初始化converter
        self.uscArab2Latn = converter.UgScriptConverter('UAS', 'ULS', True)
        self.uscLatn2Arab = converter.UgScriptConverter('ULS', 'UAS', True)

    def __call__(self, text, source_lang, target_lang):
        # translation type: same type 0, Arab-Latn 1, Arab-Zho 2, Latn-Arab 3, Latn-Zho 4, Zho-Arab 5, Zho-Latn 6
        if(source_lang == target_lang):
            self.type = 0
        elif(source_lang == 'uig_Arab' and target_lang == 'uig_Latn'):
            self.type = 1
        elif(source_lang == 'uig_Arab' and target_lang == 'zho_Hans'):
            self.type = 2
        elif(source_lang == 'uig_Latn' and target_lang == 'uig_Arab'):
            self.type = 3
        elif(source_lang == 'uig_Latn' and target_lang == 'zho_Hans'):
            self.type = 4
        elif(source_lang == 'zho_Hans' and target_lang == 'uig_Arab'):
            self.type = 5
        elif(source_lang == 'zho_Hans' and target_lang == 'uig_Latn'):
            self.type = 6
        else:
            return str("暂不支持该语种")

        if self.type == 0:
            return text
        # convert-only
        if self.type == 1:
            return self.translateArab2Latn(text)
        if self.type == 3:
            return self.translateLatn2Arab(text)

        #  nllb-only
        if self.type == 2:
            return self.translateArab2Hans(text)
        
        if self.type == 5:
            return self.translateHans2Arab(text)
        
        # convert and nllb
        if self.type == 4:
            return self.translateLatn2Hans(text)
        
        if self.type == 6:
            return self.translateHans2Latn(text)

    def translateArab2Latn(self,text):
        return self.uscArab2Latn(text)
    
    def translateArab2Hans(self,text):
        inputs_token = self.tokenizerArab2Hans(text, return_tensors="pt")
        translated_tokens = self.model.generate(**inputs_token, forced_bos_token_id=self.tokenizerArab2Hans.lang_code_to_id['zho_Hans'],max_length=self.max_length)
        return self.tokenizerArab2Hans.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    def translateLatn2Arab(self,text):
        return self.uscLatn2Arab(text)

    def translateLatn2Hans(self,text):
        inputs_token = self.tokenizerArab2Hans(self.uscLatn2Arab(text), return_tensors="pt")
        translated_tokens = self.model.generate(**inputs_token, forced_bos_token_id=self.tokenizerArab2Hans.lang_code_to_id['zho_Hans'],max_length=self.max_length)
        return self.tokenizerArab2Hans.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    def translateHans2Arab(self,text):
        inputs_token = self.tokenizerHans2Arab(text, return_tensors="pt")
        translated_tokens = self.model.generate(**inputs_token, forced_bos_token_id=self.tokenizerHans2Arab.lang_code_to_id['uig_Arab'],max_length=self.max_length)
        return self.tokenizerHans2Arab.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    def translateHans2Latn(self,text):
        inputs_token = self.tokenizerHans2Arab(text, return_tensors="pt")
        translated_tokens = self.model.generate(**inputs_token, forced_bos_token_id=self.tokenizerHans2Arab.lang_code_to_id['uig_Arab'],max_length=self.max_length)
        return self.uscArab2Latn(self.tokenizerHans2Arab.batch_decode(translated_tokens, skip_special_tokens=True)[0])
        

if __name__ == "__main__":

    # 下载模型至本地
    # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    # tokenizer_uig = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="uig_Arab")
    # model.save_pretrained("/home/disk3/lipeiyu/NLLB/data/nllb-200-distilled-600M")
    # tokenizer_uig.save_pretrained("/home/disk3/lipeiyu/NLLB/data/nllb-200-distilled-600M")

    translator = Translator("/home/disk3/lipeiyu/NLLB/data/nllb-200-distilled-600M",True,400)

    print(translator.translateArab2Latn('قول باش پۇت كۆز'))
    print(translator.translateArab2Hans('ئىيۇن» خەلقئارا بالىلار بايرىمى يېتىپ كېلىش پەيتىدە، جۇڭگو كوممۇنىستىك پارتىيەسى مەركىزىي كومىتېتىنىڭ باش شۇجىسى، دۆلەت رەئىسى، مەركىزىي ھەربىي كومىتېتنىڭ رەئىسى شى جىنپىڭ 5 - ئاينىڭ 31 - كۈنى چۈشتىن بۇرۇن بېيجىڭ يۈيىڭ مەكتىپىگە كېلىپ، ئوقۇتقۇچى - ئوقۇغۇچىلارنى يوقلىدى ۋە ئۇلاردىن ھال سورىدى، پۈتۈن مەملىكەتتىكى كەڭ ئۆسمۈر - بالىلارنىڭ بايرىمىنى تەبرىكلىدى. شى جىنپىڭ مۇنداق تەكىتلىدى: ئۆسمۈر - بالىلار ۋەتەننىڭ كېلەچىكى، جۇڭخۇا مىللىتىنىڭ ئۈمىدى. يېڭى دەۋردىكى جۇڭگو بالىلىرى ئىرادىسى بار، غايىسى بار، ئۆگىنىشنى سۆيىدىغان، ئەمگەكنى سۆيىدىغان، مىننەتدارلىقنى بىلىدىغان، دوستلۇقنى بىلىدىغان، يېڭىلىق يارىتىشقا جۈرئەت قىلىدىغان، كۈرەش قىلىشقا جۈرئەت قىلىدىغان، ئەخلاقىي، ئەقلىي، جىسمانىي، گۈزەللىك، ئەمگەك جەھەتلەردە ئەتراپلىق يېتىلگەن ياخشى بالىلاردىن بولۇشى كېرەك. ساۋاقداشلارنىڭ قۇدرەتلىك دۆلەت قۇرۇش، مىللەتنى گۈللەندۈرۈش ئۈچۈن ئوقۇشقا ئىرادە باغلاپ، ئاتا - ئانىلارنىڭ ئۈمىدىنى يەردە قويماسلىقىنى، پارتىيە ۋە خەلقنىڭ ئۈمىدىنى يەردە قويماسلىقىنى ئۈمىد قىلىمەن.'))
    print(translator.translateLatn2Arab('qol bash put köz'))
    print(translator.translateLatn2Hans('qol bash put köz'))
    print(translator.translateHans2Arab('头脚眼睛'))
    print(translator.translateHans2Latn('头脚眼睛'))