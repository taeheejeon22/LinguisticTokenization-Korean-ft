# 토큰화 방식별 토크나이저

import re
import unicodedata

from itertools import chain
from _mecab import Mecab
from soynlp.hangle import compose, decompose, character_is_korean, character_is_complete_korean, character_is_moum, character_is_jaum

doublespace_pattern = re.compile('\s+')


class tokenizers():
    def __init__(self, dummy_letter: str = "", space_symbol: str = "", grammatical_symbol: list =["", ""], nfd: bool = False):
        self.dummy_letter = dummy_letter    # 초성/중성/종성 더미 문자
        self.space_symbol = space_symbol    # 띄어쓰기 더미 문자
        self.grammatical_symbol = grammatical_symbol    # 문법 형태소 표지 # ["⭧", "⭨"]
        self.grammatical_symbol_josa = grammatical_symbol[0]    # "⫸"   # chr(11000)
        self.grammatical_symbol_eomi = grammatical_symbol[1]    # "⭧"   # chr(11111)

        self.nfd = nfd # Unicode normalization Form D

        if self.nfd == True:    # nfd 사용하면
            self.coda_normalization = False # 종성 문자는 종성 문자로 그대로 둠.  # [('나', 'NP'), ('ᆫ', 'JX')]
        elif self.nfd == False: # nfd 안 쓰면  # ㄴㅏㄴㅡㄴ
            self.coda_normalization = True  # 초성/종성 문자 통합   # [('나', 'NP'), ('ㄴ', 'JX')]


        self.mc_orig = Mecab(use_original=True)
        self.mc_fixed = Mecab(use_original=False)

        self.grammatical_pos_josa = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", "VCP"]    # 계사/지정사/서술격 조사 '이' 포함
        self.grammatical_pos_eomi = ["EP", "EF", "EC", "ETN", "ETM"]
        self.grammatical_pos = self.grammatical_pos_josa + self.grammatical_pos_eomi

        self.punctuation_pos = ["SF", "SE", "SSO", "SSC", "SC", "SY"]

        self.lexical_pos = ["NNG", "NNP", "NNB", "NNBC", "NR", "NP",
                            # "VV", "VA", "VX", "VCP", "VCN",
                            "VV", "VA", "VX", "VCN",    #     # 계사/지정사/서술격 조사 '이' 제외
                            "MM", "MAG", "MAJ", "IC",
                            "XPN", "XSN", "XSV", "XSA", "XR", ""]  # 접사도 포함시킴. 어미, 조사 아닌 것이므로.

        self.not_hangul_pos = ["SF", "SE", "SSO", "SSC", "SC", "SY", "SL", "SH", "SN"]


    ### general funcionts
    # 음절 분해용: 난 > ㄴㅏㄴ
    # https://github.com/ratsgo/embedding/blob/master/preprocess/unsupervised_nlputils.py
    def transform_v2(self, char):
        if char == ' ':  # 공백은 그대로 출력
            return char

        cjj = decompose(char)  # soynlp 이용해 분해

        # 자모 하나만 나오는 경우 처리 # ㄴ ㅠ
        try:
            if cjj.count(" ") == 2:
                if character_is_jaum(cjj[0]):  # 그 자모가 자음이면
                    cjj = (self.dummy_letter, self.dummy_letter, cjj[0])  # ('ㄴ', ' ', ' ') > ('-', 'ㄴ', '-')
                elif character_is_moum(cjj[0]):  # 그 자모가 모음이면
                    cjj = (self.dummy_letter, cjj[1], self.dummy_letter)  # (' ', 'ㅠ', ' ') > ('-', 'ㅠ', '-')
        except AttributeError:  # 혹시라도 한글 아닌 것이 들어올 경우 대비해
            pass

        if len(cjj) == 1:
            return cjj

        cjj_ = ''.join(c if c != ' ' else self.dummy_letter for c in cjj)
        return cjj_


    def transform_v3(self, char):
        if char == ' ':  # 공백은 그대로 출력
            return char

        cjj = unicodedata.normalize('NFD', char)    # unicode normalization (음절을 3개로)

        return cjj


    # 문자열 일부 치환
    def str_substitution(self, orig_str, sub_idx, sub_str):
        lst_orig_str = [x for x in orig_str]

        lst_orig_str[sub_idx] = sub_str

        return "".join(lst_orig_str)


    # for inserting space_symbol ("▃")
    # https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list
    def intersperse(self, lst, item):
        result = [[item]] * (len(lst) * 2 - 1)
        result[0::2] = lst
        return result


    # for inserting grammar_symbol ("⭧")
    def insert_grammar_symbol(self, mor_pos):
        # mor_pos: ('나', 'NP')
        # mor_pos: ('ㄴ', 'JX')
        # mor_pos: ('난', 'NP+JX')   # 무시

        pos = mor_pos[1]

        if pos in self.grammatical_pos_josa:    # 조사이면
            new_mor = self.grammatical_symbol_josa + mor_pos[0]
        elif pos in self.grammatical_pos_eomi:  # 어미이면
            new_mor = self.grammatical_symbol_eomi + mor_pos[0]
        else:   # 어휘 형태소이면
            new_mor = mor_pos[0]

        return (new_mor, pos)


    # https://github.com/ratsgo/embedding/blob/master/preprocess/unsupervised_nlputils.py
    # def str2jamo(self, sent, jamo_morpheme=False):
    def str2jamo(self, sent, grammatical=False):

        def transform_grammatical(char, grammatical):
            if char == ' ':
                return char
            cjj = decompose(char)

            if len(cjj) == 1:
                return cjj

            if grammatical == False:
                cjj_ = ''.join(c if c != ' ' else self.dummy_letter for c in cjj)
                return cjj_

            elif grammatical == True:
                cjj_without_blank = [x for x in cjj if x != " "] # remove " " from cjj

                if len(cjj_without_blank) == 1:   # if it is a jamo character (e.g. ㄴ, ㄹ, 'ㄴ'다)
                    cjj_ = self.dummy_letter * 2 + cjj_without_blank[0]

                elif len(cjj_without_blank) != 1:   # if it is a syllable character (e.g. 은, 을, 는다)
                    cjj_ = ''.join(c if c != ' ' else self.dummy_letter for c in cjj)

                return cjj_

        sent_ = []
        for char in sent:
            if character_is_korean(char):
                sent_.append(transform_grammatical(char, grammatical=grammatical))
            else:
                sent_.append(char)
        sent_ = doublespace_pattern.sub(' ', ''.join(sent_))
        return sent_


    # https://github.com/ratsgo/embedding/blob/master/models/word_eval.py 참고
    def jamo2str(self, jamo):
        jamo_list, idx = [], 0
        while idx < len(jamo):
            if jamo[idx] == self.dummy_letter:  # -ㅠ- 처리용
                jamo_list.append(jamo[idx:idx + 3])
                idx += 3

            elif not character_is_korean(jamo[idx]):  #
                jamo_list.append(jamo[idx])
                idx += 1
            else:
                jamo_list.append(jamo[idx:idx + 3])
                idx += 3


        word = ""
        for jamo_char in jamo_list:
            if len(jamo_char) == 1:
                word += jamo_char
            elif jamo_char[2] == self.dummy_letter:
                if jamo_char.count(self.dummy_letter) == 1:  # 일반적인 음절 문자 (ㅅㅏ-)
                    word += compose(jamo_char[0], jamo_char[1], " ")
                elif jamo_char.count(self.dummy_letter) == 2:  # 자모 하나만 있는 경우 (ㅋ--)
                    word += jamo_char.replace(self.dummy_letter, "")  # dummy letter 삭제하고 더하기
            elif (jamo_char[0] == self.dummy_letter) and (jamo_char[1] == self.dummy_letter):   # 문법 형태소 (--ㄹ, --ㄴ(다))
                previous_syllable = decompose(word[-1])
              
                word = word[:-1] + compose(previous_syllable[0], previous_syllable[1], jamo_char.replace(self.dummy_letter, "")) # dummy letter 삭제하고 앞 음절과 합치기


            else:
                word += compose(jamo_char[0], jamo_char[1], jamo_char[2])
        return word



    ######## tokenizer ###############
    ## 0. eojeol
    def eojeol_tokenizer(self, sent, decomposition_type: str):
                         # nfd: bool = False, morpheme_normalization: bool = False):
        # morpheme_normalization: 좋아해 -> 좋아하아


        p_multiple_spaces = re.compile("\s+")  # multiple blanks

        if decomposition_type == "composed":
        # if nfd == False:
        #     eojeol_tokenized = re.sub(p_multiple_spaces, " ", sent).split(" ")
            eojeol_tokenized = sent.split()

        elif decomposition_type == "decomposed_simple":
            if self.nfd == True:
                eojeol_tokenized = [self.transform_v3(eojeol) for eojeol in re.sub(p_multiple_spaces, " ", sent).split(" ")]
            elif self.nfd == False:
                eojeol_tokenized = [self.str2jamo(eojeol) for eojeol in re.sub(p_multiple_spaces, " ", sent).split(" ")]

        return eojeol_tokenized


    ## 1. morpheme
    def mecab_tokenizer(self, sent: str, token_type: str, tokenizer_type: str, decomposition_type: str, flatten: bool = True, lexical_grammatical: bool = False):
        assert (tokenizer_type in ["none", "mecab_orig", "mecab_fixed"] ), 'check the tokenizer type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

        # eojeol tokenization
        if token_type == "eojeol":
            mecab_tokenized = self.eojeol_tokenizer(sent=sent, decomposition_type=decomposition_type)

        # morpheme tokenization
        elif token_type == "morpheme":
            if tokenizer_type == "mecab_orig":
                use_original = True
            elif tokenizer_type == "mecab_fixed":
                use_original = False

            if decomposition_type == "composed":
                mecab_tokenized = self.mecab_composed_decomposed_simple(sent=sent, lexical_grammatical=lexical_grammatical, use_original=use_original, simple_decomposition=False, nfd=self.nfd, flatten=flatten)

            elif decomposition_type == "decomposed_simple":
                mecab_tokenized = self.mecab_composed_decomposed_simple(sent=sent, lexical_grammatical=lexical_grammatical, use_original=use_original, simple_decomposition=True, nfd=self.nfd, flatten=flatten)

            elif decomposition_type == "decomposed_lexical":
                mecab_tokenized = self.mecab_with_morphological_decomposition(sent=sent, lexical_grammatical=lexical_grammatical, use_original=use_original, nfd=self.nfd, lexical_or_grammatical="lexical", flatten=flatten)

            elif decomposition_type == "decomposed_grammatical":
                mecab_tokenized = self.mecab_with_morphological_decomposition(sent=sent, lexical_grammatical=lexical_grammatical, use_original=use_original, nfd=self.nfd, lexical_or_grammatical="grammatical", flatten=flatten)

        return mecab_tokenized


    # 1-1. composed & decomposed_simple
    def mecab_composed_decomposed_simple(self, sent: str, use_original: bool, simple_decomposition: bool, nfd: bool, flatten: bool = True, lexical_grammatical: bool = False):
        # 문법 형태소만 분리: 육식동물 에서 는
        if lexical_grammatical == True:
            mor_poss = self.mecab_grammatical_tokenizer(sent=sent)

        # 순수 형태소 분석: 육식 동물 에서 는
        elif lexical_grammatical == False:
            if use_original == True:
                mor_poss = self.mc_orig.pos(sent, flatten=False, coda_normalization=self.coda_normalization)  # [[('넌', 'NP+JX')], [('날', 'NNG')], [('좋', 'VA'), ('아', 'EC'), ('해', 'VV+EC')]]

            else:
                mor_poss = self.mc_fixed.pos(sent, flatten=False, coda_normalization=self.coda_normalization)  # [[('너', 'NP'), ('ㄴ', 'JX')], [('날', 'NNG')], [('좋', 'VA'), ('아', 'EC'), ('하', 'VV'), ('아', 'EC')]]


        # insert grammatical symbol
        if len(self.grammatical_symbol) > 0:   # grammatical_symbol 사용하면
            mor_poss = [[self.insert_grammar_symbol(mor_pos=mor_pos) for mor_pos in word] for word in mor_poss]


        # remove pos tags
        if simple_decomposition == False:
            mors = [[mor_pos[0] for mor_pos in word] for word in mor_poss]  # [['너', 'ㄴ'], ['날'], ['좋', '아', '하', '아']]

        elif simple_decomposition == True:
            if nfd == False:
                mors = [ [ self.str2jamo(mor_pos[0], grammatical=True)  if (mor_pos[-1] in self.grammatical_pos ) else self.str2jamo(mor_pos[0], grammatical=False) for mor_pos in word] for word in mor_poss]
                                                                        # convert jamo morpheme like ㄴ, ㄹ into ##ㄴ, ##ㄹ
            elif nfd == True:
                mors = [[self.transform_v3(mor_pos[0]) for mor_pos in word] for word in mor_poss]


        if flatten == True:
            mecab_tokenized = list(chain.from_iterable(self.intersperse(mors, self.space_symbol)))  # ['너', 'ㄴ', '▃', '날', '▃', '좋', '아', '하', '아']

            if self.space_symbol == "": # 스페이스 심벌 안 쓴다면
                mecab_tokenized = [token for token in mecab_tokenized if token != ""]    # 빈 토큰 제외


        elif flatten == False:
            mecab_tokenized = self.intersperse(mors, self.space_symbol)  # [['너', 'ㄴ'], ['▃'], ['날'], ['▃'], ['좋', '아', '하', '아']]

            if self.space_symbol == "": # 스페이스 심벌 안 쓴다면
                mecab_tokenized = [token for token in mecab_tokenized if token != [""]]    # 빈 토큰 제외

        return mecab_tokenized



    ## 1-2. decomposition morphological
    def mecab_with_morphological_decomposition(self, sent: str, use_original: bool, nfd: bool = False, lexical_or_grammatical: str = "lexical", flatten: bool = True, lexical_grammatical: bool = False):
        '''
        :param sent: 자모 변환할 문장      '너를 좋아해'
        :param morpheme_analysis:
            False: 자모 변환만 수행    (어절 토큰화 문장을 자모로 변환하는 데에 그대로 이용 가능)
            True: 형태소 분석 + 자모 변환
        :param use_original: konlpy original mecab 쓸지
        :param nfd: unicode NFD 적용해서 분해할지.
        :return: 자모 변환된 문장          '너ㅡㄹ 좋아해' or '너 ㄹㅡㄹ 좋아해'
        '''

        # 문법 형태소만 분리: 육식동물 에서 는
        if lexical_grammatical == True:
            mor_poss = self.mecab_grammatical_tokenizer(sent=sent)

        # 순수 형태소 분석: 육식 동물 에서 는
        elif lexical_grammatical == False:
            if use_original == True:
                mor_poss = self.mc_orig.pos(sent, flatten=False, coda_normalization=self.coda_normalization)  # 형태소 분석
                    
            elif use_original == False:
                mor_poss = self.mc_fixed.pos(sent, flatten=False, coda_normalization=self.coda_normalization)  # 형태소 분석


        new_sent = list()
        for ix in range(len(mor_poss)):
            eojeol = mor_poss[ix]  # [('나', 'NP'), ('는', 'JX')]

            new_eojeol = list()  # ['나', 'ㄴㅡㄴ']
            for jx in range(len(eojeol)):
                morpheme, pos = eojeol[jx]  # '너', 'NP'

                if lexical_or_grammatical == "lexical":
                    if pos in self.grammatical_pos_josa: # 조사이면
                        decomposed_morpheme = self.grammatical_symbol_josa + morpheme[:]
                    elif pos in self.grammatical_pos_eomi: # 어미이면
                        decomposed_morpheme = self.grammatical_symbol_eomi + morpheme[:]

                    else: # 어휘 형태소 혹은 혼종('난/NP+JX')이면
                        if nfd == False:
                            decomposed_morpheme = "".join(
                                [self.transform_v2(char) if character_is_korean(char) else char for char in morpheme])  # 한 -> ㅎㅏㄴ
                        elif nfd == True:
                            decomposed_morpheme = "".join(
                                [self.transform_v3(char) if character_is_korean(char) else char for char in morpheme])  # 는 -> 는  # len("는"): 3

                elif lexical_or_grammatical == "grammatical":
                    # 문법 형태소가 아니면
                    if not (pos in self.grammatical_pos):
                        decomposed_morpheme = morpheme[:]

                    else:   # 문법 형태소이면

                        if pos in self.grammatical_pos_josa:   # 조사이면
                            grammatical_symbol = self.grammatical_symbol_josa[:]
                        elif pos in self.grammatical_pos_eomi:   # 어미이면
                            grammatical_symbol = self.grammatical_symbol_eomi[:]

                        if nfd == False:
                            decomposed_morpheme = grammatical_symbol + "".join(
                                [self.transform_v2(char) if character_is_korean(char) else char for char in morpheme])  # 한 -> ㅎㅏㄴ
                        elif nfd == True:
                            decomposed_morpheme = grammatical_symbol + "".join(
                                [self.transform_v3(char) if character_is_korean(char) else char for char in morpheme])  # 는 -> 는  # len("는"): 3

                new_eojeol.append(decomposed_morpheme)

            new_sent.append(new_eojeol)

        if flatten == True:
            new_sent_with_special_token = list(chain.from_iterable(self.intersperse(new_sent, self.space_symbol)))    # ['너', 'ㄴ', '▃', '날', '▃', '좋', '아', '하', '아']

            if self.space_symbol == "": # 스페이스 심벌 안 쓴다면
                new_sent_with_special_token = [token for token in new_sent_with_special_token if token != ""]    # 빈 토큰 제외

        elif flatten == False:
            new_sent_with_special_token = self.intersperse(new_sent, self.space_symbol)  # [['너', 'ㄴ'], ['▃'], ['날'], ['▃'], ['좋', '아', '하', '아']]

            if self.space_symbol == "":  # 스페이스 심벌 안 쓴다면
                new_sent_with_special_token = [token for token in new_sent_with_special_token if token != [""]]  # 빈 토큰 제외

        return new_sent_with_special_token


    # 어절에서 조사와 어미만 분리하기 # 한국초등학교에서는 -> 육식동물/LEXICAL, '에서/JKB', '는/JX'
    def mecab_grammatical_tokenizer(self, sent: str):

        mors_ejs_in_sent = self.mc_fixed.pos(sent, flatten=False, coda_normalization=self.coda_normalization)  # 형태소 분석


        def concat_lexical_morphemes(eojeol: list):
            lexical_start_idxs = list()
            lexical_end_idxs =list()

            for idx, cnt in enumerate(eojeol):
                if idx != 0:
                    if (cnt[-1] in self.lexical_pos) and (not eojeol[idx - 1][-1] in self.lexical_pos):  # 어휘 형태소이고, 이전 형태소가 어휘 형태소가 아니라면(문법 형태소 or 기호라면)
                        lexical_start_idxs.append(idx)
                    elif (not cnt[-1] in self.lexical_pos) and (eojeol[idx - 1][-1] in self.lexical_pos):  # 어휘 형태소가 아니고(문법 형태소 or 기호이고), 이전 형태소가 어휘 형태소이면
                        lexical_end_idxs.append(idx)

                    if idx == len(eojeol) - 1:  # 어절 마지막 형태소이면

                        if (len(lexical_start_idxs)-1 == len(lexical_end_idxs)) and lexical_start_idxs[-1] < idx + 1:    # lexical_start_idx가 lexical_end_idx보다 하나 많고, 현재 인덱스+1보다 이전에 lexical_start_idx가 있으면
                            lexical_end_idxs.append(idx+1)

                elif idx == 0:
                    if (cnt[-1] in self.lexical_pos):  # 어휘 형태소이면
                        lexical_start_idxs.append(idx)

                        if len(eojeol)==1:  # 어휘 형태소 하나로만 된 어절이면
                            lexical_end_idxs.append(idx+1)  # end_idx까지 추가

            assert(len(lexical_start_idxs)==len(lexical_end_idxs)), ("concat_lexical_morphemes ERROR!!!")    # 무결성 검사

            lexical_idxs = list(zip(lexical_start_idxs, lexical_end_idxs))    # 어절 내 어휘 형태소의 (시작, 끝) 인덱스의 list

            new_eojeol = eojeol[:]

            for jx in range(len(lexical_idxs)):
                concat_eojeol_proto = [ "DUMMY" ] * (lexical_idxs[jx][-1]-lexical_idxs[jx][0])  # 띄어쓰기 안 돼서 한 어절 내에 LEXICAL 여러 번 나타나는 경우 대비해, 인덱스 유지하기 위한 DUMMY 형태소 추가
                concat_eojeol_proto[0] = ( "".join( [mor_pos[0] for mor_pos in eojeol[lexical_idxs[jx][0]:lexical_idxs[jx][-1]] ] ), "LEXICAL" )

                new_eojeol[lexical_idxs[jx][0]:lexical_idxs[jx][-1]] = concat_eojeol_proto

            new_eojeol = [mor_pos for mor_pos in new_eojeol if mor_pos != "DUMMY"]  # DUMMY 형태소 제거

            return new_eojeol

        new_sent = list()

        for ix in range(len(mors_ejs_in_sent)):
            eojeol = mors_ejs_in_sent[ix]

            new_eojeol = concat_lexical_morphemes(eojeol=eojeol)

            new_sent.append(new_eojeol)

        return new_sent



## 자모 분해된 것을 원래 문장으로 복원. 완벽하지 않음.
    def jamo2str_morphological(self, jamo):
        jamo_eojeols = jamo.split(" ")  # ['나는', '즐겁게', '밥을', '먹는다.']

        # 각 토큰을 음절 단위로 분해
        eojeols_not_composed = list()   # [['나', 'ㄴㅡㄴ'], ['즐', '겁', 'ㄱㅔ-'], ['밥', 'ㅇㅡㄹ'], ['먹', 'ㄴㅡㄴ', 'ㄷㅏ-', '.']]

        for ix in range(len(jamo_eojeols)):
            jamo_eojeol = jamo_eojeols[ix]      # '나ㄴㅡㄴ'

            composed_eojeol, idx = list(), 0    # ['나', 'ㄴㅡㄴ']
            while idx < len(jamo_eojeol):

                if character_is_complete_korean(jamo_eojeol[idx]):  # 이미 합쳐져 있는 음절 문자이면
                    composed_eojeol.append(jamo_eojeol[idx])
                    idx += 1

                elif character_is_jaum(jamo_eojeol[idx]):   # 분해된 자모이면
                    composed_eojeol.append(jamo_eojeol[idx:idx + 3])
                    idx += 3

                else:   # 기타: 특수 기호 등
                    composed_eojeol.append(jamo_eojeol[idx])
                    idx += 1

            eojeols_not_composed.append(composed_eojeol)


        composed_str = list()   # ['나는', '괜찮아']
        for jx in range(len(eojeols_not_composed)):
            eojeol_not_composed = eojeols_not_composed[jx]  # ['나', 'ㄴㅡㄴ'],

            eojeol_composed = str()
            for kx in range(len(eojeol_not_composed)):
                char_not_composed = eojeol_not_composed[kx]

                if len(char_not_composed) == 3 and sum([1 for char in char_not_composed if character_is_jaum(char)]) >= 1:  # 합쳐야 되는 문자열이면
                    if not self.dummy_letter in char_not_composed:
                        eojeol_composed += compose(char_not_composed[0], char_not_composed[1], char_not_composed[2])
                    elif self.dummy_letter in char_not_composed:
                        if char_not_composed.count(self.dummy_letter) == 1:   # ㅅㅏ- 등의 일반적인 경우
                            eojeol_composed += compose(char_not_composed[0], char_not_composed[1], " ")
                        elif char_not_composed.count(self.dummy_letter) == 2:   # ㄴ-- (조사 'ㄴ') 등의 경우

                            if kx != 0: # 앞 음절이 있으면
                                chosung, junsung, jonsung = list(decompose(eojeol_not_composed[kx-1])[:2]) + [eojeol_not_composed[kx].replace(self.dummy_letter, "")] # 앞 음절 분해 후 붙이기
                                eojeol_composed = eojeol_composed[:kx-1]    # 앞 음절을 eojeol_composed에서 삭제
                                eojeol_composed += compose(chosung, junsung, jonsung)  # 앞 음절에 붙여서 합성   나 + ㄴ -> 난
                            elif kx == 0:   # 앞 음절이 없으면
                                eojeol_composed += char_not_composed

                else:   # 합치지 말아야 될 문자열이면 (이미 합쳐져 있는 음절 문자, 특수 기호 등)
                    eojeol_composed += char_not_composed

            composed_str.append(eojeol_composed)


        # 합성 안 된 것이 있는지 체크
        not_composed_idx =  [ix for ix, token in enumerate(composed_str) if self.dummy_letter in token]
        if len(not_composed_idx) >=1 :  # ['나', 'ㄴ--', '괜찮', '아'] 같은 경우
            for lx in range(len(not_composed_idx)):
                if not_composed_idx[lx] >= 1:
                    previous_token = composed_str[not_composed_idx[lx]-1] # 바로 앞 토큰

                    if decompose(previous_token[-1])[-1] == ' ':    # 앞 음절(바로 앞 토큰의 마지막 음절)이 종성이 없다면
                        chosung, junsung, jonsung = list(decompose(previous_token[-1])[:2]) + [composed_str[ not_composed_idx[lx] ].replace(self.dummy_letter, "")]

                        new_previous_token = self.str_substitution(orig_str=previous_token, sub_idx=-1, sub_str=compose(chosung, junsung, jonsung))
                        composed_str[not_composed_idx[lx]-1] = new_previous_token[:]    # 앞 토큰을 새로 합성한 것으로 치환   # '나' > '난'
                        composed_str[not_composed_idx[lx]] = ""  # 현재 토큰은 공백으로 치환

                    elif decompose(previous_token[-1])[-1] != " ":    # 앞 음절(바로 앞 토큰의 마지막 음절)이 종성이 있다면
                        composed_str[not_composed_idx[lx]] = composed_str[not_composed_idx[lx]].replace(self.dummy_letter, "")    # 그냥 coda letter만 삭제하기

                elif not_composed_idx[lx] == 0:
                        composed_str[not_composed_idx[lx]] = composed_str[not_composed_idx[lx]].replace(self.dummy_letter, "")    # 그냥 coda letter만 삭제하기

        composed_str = [token for token in composed_str if token != ""]

        return " ".join(composed_str)   # "나는 괜찮아"
