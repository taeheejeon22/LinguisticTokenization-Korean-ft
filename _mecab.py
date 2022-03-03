# based on KoNLPy _mecab.py

# requirements
    # KoNLPy 0.5.2
    # mecab-0.996-ko-0.9.2  /   mecab-ko-dic: 2.1.1

# how to use
''' python
from _mecab import Mecab

mc = Mecab(use_original=True)   # use_original: True(use KoNLPy version), False(use our version)

'''
#

# -*- coding: utf-8 -*-
from __future__ import absolute_import

import itertools     # for list flattening
import re            # for clearing unnecessary attrs (e.g. 불태워/VV/*, 터/NNP/인명)

import sys

try:
    from MeCab import Tagger
except ImportError:
    pass



from konlpy import utils


__all__ = ['Mecab']


attrs = ['tags',        # 품사 태그
         'semantic',    # 의미 부류
         'has_jongsung',  # 종성 유무
         'read',        # 읽기
         'type',        # 타입
         'first_pos',   # 첫 번째 품사
         'last_pos',    # 마지막 품사
         'original',    # 원형
         'indexed']     # 인덱스 표현

regexp = re.compile(".+(?=/[^A-Z])") # a pattern for only morphemes and their POS (e.g. 불태워/VV/* > 불태워/VV)


######################## the original code ##############################
def parse(result, allattrs=False, join=False):
    def split(elem, join=False):
        if not elem: return ('', 'SY')
        s, t = elem.split('\t')
        if join:
            return s + '/' + t.split(',', 1)[0]
        else:
            return (s, t.split(',', 1)[0])

    return [split(elem, join=join) for elem in result.splitlines()[:-1]]
#########################################################################


# e.g. 이게 뭔지 알아.
# >
# [('이것', 'NP'),
# ('이', 'JKS'),
# ('뭐', 'NP'),
# ('이', 'VCP'),
# ('ㄴ지', 'EC'),
# ('알', 'VV'),
# ('아', 'EF'),
# ('.', 'SF')])


# new functions
# function for getting the (morpheme, POS) list of a sentence
# output of parse()       (original): [('너', 'NP'), ('를', 'JKO'), ('좋아해', 'VV+EF'), ('.', 'SF')]
# output of parse_fixed() (fixed)   : [('너', 'NP'), ('를', 'JKO'), ('좋아하', 'VV'), ('아', 'EF'), ('.', 'SF')]
def parse_fixed(result, allattrs=False, join=False):
        # result: an analysed result of a sentence (e.g. 이게 뭔지 알아. > 이게\tNP+JKS,*,F,이게,Inflect,NP,JKS,이것/NP/*+이/JKS/*\n뭔지\tNP+VCP+EC,*,F,뭔지,Inflect,NP,EC,뭐/NP/*+이/VCP/*+ㄴ지/EC/*\n알\tVV,*,T,알,*,*,*,*\n아\tEF,*,F,아,*,*,*,*\n.\tSF,*,*,*,*,*,*,*\nEOS\n)

    def split(elem, join=False):
            # elem: an analysed result of an eojeol (e.g. 뭔지 > 뭔지\tNP+VCP+EC,*,F,뭔지,Inflect,NP,EC,뭐/NP/*+이/VCP/*+ㄴ지/EC/*)

        if not elem:
            if join == False:
                return ('', 'SY')
            elif join == True:
                return '/SY'

        s, t = elem.split('\t') # s: an eojeol (e.g. 위한)   # t: analysed resulf of an eojeol (e.g. VV+ETM,*,T,위한,Inflect,VV,ETM,위하/VV/*+ᆫ/ETM/*)
        token_pos = t.split(',')[0] # original token POS of mecab-ko (e.g. 위한: VV+ETM)
        lst_morpos = t.split(',')[-1].split("+")  # splitting the last attr (인덱스 표현) of 't' by morpheme (e.g. 위하/VV/*+ᆫ/ETM/* > ["위하/VV/*", "ᆫ/ETM/*"])

        if join:
            if not t.split(',')[4].startswith("Inflect"): # If an eojeol is not Inflect (= a concatenation of morphemes is equal to its original eojeol. e.g. 해수욕장 == 해수 + 욕 + 장)
                return s + '/' + token_pos  # eojeol + / + POS (e.g. 위한/VV+ETM)

            else:   # If an eojeol is Inflect (= a concatenation of morphemes is not equal to its original eojeol) (e.g. 불태워졌다 != 불태우 + 어 + 지 + 었 + 다)
                mor_info = [regexp.search(x).group() for x in lst_morpos] # make a list of morphemes with their POSs (e.g. ['줍/VV', '어서/EC'])

                # There is a bug that outputs of mecab-ko-dic are different according to OS, and OS versions. This is a make-shift.
                if len(mor_info) > 1:
                    return mor_info
                elif len(mor_info) == 1:
                    return [s + "/" + token_pos]
                # return [regexp.search(x).group() for x in lst_morpos]   # make a list of morphemes with their POSs (e.g. ['줍/VV', '어서/EC'] )

        else:
            if not t.split(',')[4].startswith("Inflect"):
                return (s, token_pos)

            else:
                mor_info = [tuple(regexp.search(x).group().split("/")) for x in lst_morpos] # make a list of morphemes with their POSs (e.g. [('줍', 'VV'), ('어서', 'EC')] )

                # There is a bug that outputs of mecab-ko-dic are different according to OS, and OS versions. This is a make-shift.
                if len(mor_info) > 1:
                    return mor_info
                elif len(mor_info) == 1:
                    return (s, token_pos)

                # return [tuple(regexp.search(x).group().split("/")) for x in lst_morpos]


    return list ( itertools.chain.from_iterable( [ [x] if type(x) != list else x  for x in [split(elem, join=join) for elem in result.splitlines()[:-1]] ] ) )
    #                                             # making a 3-D list: [ [ (morpheme, POS), (morpheme, POS), ... ], ... ]
    #             # itertools: flattening the list to a 2-D list: [ (morpheme, POS), (morpheme, POS), ... ]


# function for multiple replacing   # https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
def replace_multiple(string, replace_list):
    # replace_tuples: [("brown", "red"), ("lazy", "quick")]
    for r in (replace_list):
        string = string.replace(*r)
    return string


# unicode error correction      # (타당하 + ㄴ지  vs. 뭐 + 이 +  ᆫ지) ->  ᆫ지    # "ᆼ" -> "ㅇ"
def hangul_unicode_correction(parsed: str):
    result_split_n = parsed.split("\n")[:-2]  # remove 'EOS', ''

    result_corrected = [token_analysis.split(",")[0] + "," + ",".join(
        [replace_multiple(string=info, replace_list=[("ㄴ", "ᆫ"), ("ㄹ", "ᆯ"), ("ㅁ", "ᄆ"), ("ㅂ", "ᄇ"), ("ᆼ", "ㅇ")]) for
         info in token_analysis.split(",")[1:]]) for token_analysis in result_split_n]

    result_corrected_final = "\n".join(result_corrected + parsed.split("\n")[-2:])

    return result_corrected_final


######################## the original code ##############################
# class Mecab():
#     """Wrapper for MeCab-ko morphological analyzer.

#     `MeCab`_, originally a Japanese morphological analyzer and POS tagger
#     developed by the Graduate School of Informatics in Kyoto University,
#     was modified to MeCab-ko by the `Eunjeon Project`_
#     to adapt to the Korean language.

#     In order to use MeCab-ko within KoNLPy, follow the directions in
#     :ref:`optional-installations`.

#     .. code-block:: python
#         :emphasize-lines: 1

#         >>> # MeCab installation needed
#         >>> from konlpy.tag import Mecab
#         >>> mecab = Mecab()
#         >>> print(mecab.morphs(u'영등포구청역에 있는 맛집 좀 알려주세요.'))
#         ['영등포구', '청역', '에', '있', '는', '맛집', '좀', '알려', '주', '세요', '.']
#         >>> print(mecab.nouns(u'우리나라에는 무릎 치료를 잘하는 정형외과가 없는가!'))
#         ['우리', '나라', '무릎', '치료', '정형외과']
#         >>> print(mecab.pos(u'자연주의 쇼핑몰은 어떤 곳인가?'))
#         [('자연', 'NNG'), ('주', 'NNG'), ('의', 'JKG'), ('쇼핑몰', 'NNG'), ('은', 'JX'), ('어떤', 'MM'), ('곳', 'NNG'), ('인가', 'VCP+EF'), ('?', 'SF')]

#     :param dicpath: The path of the MeCab-ko dictionary.

#     .. _MeCab: https://code.google.com/p/mecab/
#     .. _Eunjeon Project: http://eunjeon.blogspot.kr/
#     """

#     # TODO: check whether flattened results equal non-flattened
#     def pos(self, phrase, flatten=True, join=False):
#         """POS tagger.

#         :param flatten: If False, preserves eojeols.
#         :param join: If True, returns joined sets of morph and tag.
#         """

#         if sys.version_info[0] < 3:
#             phrase = phrase.encode('utf-8')
#             if flatten:
#                 result = self.tagger.parse(phrase).decode('utf-8')
#                 return parse(result, join=join)
#             else:
#                 return [parse(self.tagger.parse(eojeol).decode('utf-8'), join=join)
#                         for eojeol in phrase.split()]
#         else:
#             if flatten:
#                 result = self.tagger.parse(phrase)
#                 return parse(result, join=join)
#             else:
#                 return [parse(self.tagger.parse(eojeol), join=join)
#                         for eojeol in phrase.split()]

#     def morphs(self, phrase):
#         """Parse phrase to morphemes."""

#         return [s for s, t in self.pos(phrase)]

#     def nouns(self, phrase):
#         """Noun extractor."""

#         tagged = self.pos(phrase)
#         return [s for s, t in tagged if t.startswith('N')]

#     def __init__(self, dicpath='/usr/local/lib/mecab/dic/mecab-ko-dic'):
#         self.dicpath = dicpath
#         try:
#             self.tagger = Tagger('-d %s' % dicpath)
#             self.tagset = utils.read_json('%s/data/tagset/mecab.json' % utils.installpath)
#         except RuntimeError:
#             raise Exception('The MeCab dictionary does not exist at "%s". Is the dictionary correctly installed?\nYou can also try entering the dictionary path when initializing the Mecab class: "Mecab(\'/some/dic/path\')"' % dicpath)
#         except NameError:
#             raise Exception('Install MeCab in order to use it: http://konlpy.org/en/latest/install/')

#     def __setstate__(self, state):
#         """just reinitialize."""

#         self.__init__(dicpath=state['dicpath'])

#     def __getstate__(self):
#         """store arguments."""

#         return {'dicpath': self.dicpath}
#########################################################################



class Mecab():
    """Wrapper for MeCab-ko morphological analyzer.

    `MeCab`_, originally a Japanese morphological analyzer and POS tagger
    developed by the Graduate School of Informatics in Kyoto University,
    was modified to MeCab-ko by the `Eunjeon Project`_
    to adapt to the Korean language.

    In order to use MeCab-ko within KoNLPy, follow the directions in
    :ref:`optional-installations`.

    .. code-block:: python
        :emphasize-lines: 1

        >>> # MeCab installation needed
        >>> from konlpy.tag import Mecab
        >>> mecab = Mecab()
        >>> print(mecab.morphs(u'영등포구청역에 있는 맛집 좀 알려주세요.'))
        ['영등포구', '청역', '에', '있', '는', '맛집', '좀', '알려', '주', '세요', '.']
        >>> print(mecab.nouns(u'우리나라에는 무릎 치료를 잘하는 정형외과가 없는가!'))
        ['우리', '나라', '무릎', '치료', '정형외과']
        >>> print(mecab.pos(u'자연주의 쇼핑몰은 어떤 곳인가?'))
        [('자연', 'NNG'), ('주', 'NNG'), ('의', 'JKG'), ('쇼핑몰', 'NNG'), ('은', 'JX'), ('어떤', 'MM'), ('곳', 'NNG'), ('인가', 'VCP+EF'), ('?', 'SF')]

    :param dicpath: The path of the MeCab-ko dictionary.

    .. _MeCab: https://code.google.com/p/mecab/
    .. _Eunjeon Project: http://eunjeon.blogspot.kr/
    """


    # TODO: check whether flattened results equal non-flattened
    def pos(self, phrase, flatten=True, join=False, coda_normalization=True):
        if self.use_original == False:  # If we use the fixed version
            """POS tagger.

            :param flatten: If False, preserves eojeols.
            :param join: If True, returns joined sets of morph and tag.
            """

            # replacing for exceptions
            # phrase = phrase.replace('\u3000', ' ')  # replacing ideographic spaces into blanks
            # phrase = phrase.replace('영치기 영차', '영치기영차')   # a temporary solution for '영치기 영차'. '영치기 영차' consists of 2 eojeols. However, MeCab-ko analyses it as 1 eojeol. I haven't figured out the reason yet.
            phrase = replace_multiple(string=phrase, replace_list=[('\u3000', ' '), ('영치기 영차', '영치기영차')])


            # self = Mecab()
            if sys.version_info[0] >= 3: # for Python 3
                result = self.tagger.parse(phrase)  # an analysed result of a phrase (or a sentence) (e.g. 이게 뭔지 알아. > 이게\tNP+JKS,*,F,이게,Inflect,NP,JKS,이것/NP/*+이/JKS/*\n뭔지\tNP+VCP+EC,*,F,뭔지,Inflect,NP,EC,뭐/NP/*+이/VCP/*+ㄴ지/EC/*\n알\tVV,*,T,알,*,*,*,*\n아\tEF,*,F,아,*,*,*,*\n.\tSF,*,*,*,*,*,*,*\nEOS\n)


                if flatten: # flatten = True. If you want to get a flattened (2-D) result: [(morpheme, POS), ...]
                                # e.g.
                                # [('이것', 'NP'),
                                # ('이', 'JKS'),
                                # ('뭐', 'NP'),
                                # ('이', 'VCP'),
                                # ('ㄴ지', 'EC'),
                                # ('알', 'VV'),
                                # ('아', 'EF'),
                                # ('.', 'SF')])

                    # converting final consonant characters to ordinary single characters
                    # result = result.replace("ᆯ", "ㄹ").replace("ᆫ", "ㄴ").replace("ᄇ", "ㅂ").replace("ᆼ", "ㅇ")


                    result = hangul_unicode_correction(parsed=result)


                    if coda_normalization == False:
                        pass
                    elif coda_normalization == True:
                        result = replace_multiple(string=result, replace_list=[("ᆫ", "ㄴ"), ("ᆯ", "ㄹ"), ("ᄆ", "ㅁ"), ("ᄇ", "ㅂ"), ("ᆼ", "ㅇ")])

                    return parse_fixed(result, join=join)

                else:   # flatten = False. If you want to get a 3-D result: [ [ (morpheme, POS), (morpheme, POS), ... ], ... ]
        #                     # e.g.
        #                     # [[('이것', 'NP'), ('이', 'JKS')],
        #                     # [('뭐', 'NP'), ('이', 'VCP'), ('ᆫ지', 'EC')],
        #                     # [('알', 'VV'), ('아', 'EF'), ('.', 'SF')]]
        #
                    ## 1) analysed result of Mecab-ko
                    result_mor_lst = result.splitlines()[:-1]
                    # example of 'result_mor_lst'
                    # ['이게\tNP+JKS,*,F,이게,Inflect,NP,JKS,이것/NP/*+이/JKS/*',
                    # '뭔지\tNP+VCP+EC,*,F,뭔지,Inflect,NP,EC,뭐/NP/*+이/VCP/*+ㄴ지/EC/*',
                    # '알\tVV,*,T,알,*,*,*,*',
                    # '아\tEF,*,F,아,*,*,*,*',
                    # '.\tSF,*,*,*,*,*,*,*']


                    # troubleshooting an unanalyzable character: 
                    result_mor_lst = [x if x != "" else '\tSY,*,*,*,*,*,*,*' for x in result_mor_lst ]


                    ## 2) adding indices of eojeols to result_mor_lst
                    phrase2ej = phrase.split()  # eojeol list # ['먹을', '수', '있다']
                    cnt = 0 # index of an eojeol
                    concat_mor = ""

                    for i in range(len(result_mor_lst)):
                        info_str = result_mor_lst[i].split("\t")[0].strip() # '너\tNP,*,F,너,*,*,*,*'   >   '너'

                        concat_mor += info_str  # concatenating morphemes until the string is equal to their original eojeol (e.g. 알 > 알+았 > 알+았+어요)

                        if concat_mor == phrase2ej[cnt]:    # If the string of concatenated morphemes is equal to its original eojeol
                            result_mor_lst[i] += "," + str(cnt) # adding the index (cnt) of the eojeol
                            cnt += 1
                            concat_mor = ""
                        else:
                            result_mor_lst[i] += "," + str(cnt) # adding the index (cnt) of the eojeol
                    # example of 'result_mor_lst'
                    # ['이게\tNP+JKS,*,F,이게,Inflect,NP,JKS,이것/NP/*+이/JKS/*,0',   # add an eojeol index (',0')
                    # '뭔지\tNP+VCP+EC,*,F,뭔지,Inflect,NP,EC,뭐/NP/*+이/VCP/*+ㄴ지/EC/*,1',
                    # '알\tVV,*,T,알,*,*,*,*,2',
                    # '아\tEF,*,F,아,*,*,*,*,2',
                    # '.\tSF,*,*,*,*,*,*,*,2']


                    ## 3) splitting the result_mor_lst by morpheme for the case when the string of concatenated morphmese is not equal to their original eojeol (e.g. 뭔지 != 뭐 + 이 + ㄴ지)
                    for i in range(len(result_mor_lst)):
                        splited = result_mor_lst[i].split(",")

                        if splited[4] == 'Inflect': # If an eojeol is Inflect (= a concatenation of morphemes is not equal to its original eojeol. (e.g. 뭔지 != 뭐 + 이 + ㄴ지)
                            mors = [x.split('/')[0] for x in splited[7].split('+')]
                            for_replace = list()

                            for j in range(len(mors)):
                                for_replace += [ mors[j] + '\t' + result_mor_lst[i].split('\t')[-1]]

                            result_mor_lst[i] = for_replace
        # split a token anlaysis string by morpheme
        # '좋아해\tVV+EF,*,F,좋아해,Inflect,VV,EF,좋아하/VV/*+아/EF/*,1'   >    # ['좋아하\tVV+EF,*,F,좋아해,Inflect,VV,EF,좋아하/VV/*+아/EF/*,1',
        #                                                                 # '아\tVV+EF,*,F,좋아해,Inflect,VV,EF,좋아하/VV/*+아/EF/*,1']

                        else:   # If an eojeol consists of 1 morpheme
                            result_mor_lst[i] = [result_mor_lst[i]] # convert to a list (e.g. '너\tNP,*,F,너,*,*,*,*,0'  >  ['너\tNP,*,F,너,*,*,*,*,0'])

                    # flatten the list of lists to a 1-D list
                    result_mor_lst = list(itertools.chain.from_iterable(result_mor_lst))

                    # example of result_mor_lst
                    # ['이게\tNP+JKS,*,F,이게,Inflect,NP,JKS,이것/NP/*+이/JKS/*,0',                    ['이것\tNP+JKS,*,F,이게,Inflect,NP,JKS,이것/NP/*+이/JKS/*,0',
                    #                                                                             '이\tNP+JKS,*,F,이게,Inflect,NP,JKS,이것/NP/*+이/JKS/*,0',
                    # '뭔지\tNP+VCP+EC,*,F,뭔지,Inflect,NP,EC,뭐/NP/*+이/VCP/*+ㄴ지/EC/*,1',            '뭐\tNP+VCP+EC,*,F,뭔지,Inflect,NP,EC,뭐/NP/*+이/VCP/*+ㄴ지/EC/*,1',
                    #                                                                         >     '이\tNP+VCP+EC,*,F,뭔지,Inflect,NP,EC,뭐/NP/*+이/VCP/*+ㄴ지/EC/*,1',
                    #                                                                             'ㄴ지\tNP+VCP+EC,*,F,뭔지,Inflect,NP,EC,뭐/NP/*+이/VCP/*+ㄴ지/EC/*,1',
                    # '알\tVV,*,T,알,*,*,*,*,2',                                                     '알\tVV,*,T,알,*,*,*,*,2',
                    # '아\tEF,*,F,아,*,*,*,*,2',                                                     '아\tEF,*,F,아,*,*,*,*,2',
                    # '.\tSF,*,*,*,*,*,*,*,2']                                                      '.\tSF,*,*,*,*,*,*,*,2']


                    ## 4) saving the 3-D (unflattened) result:    [ [ (morpheme, POS), (morpheme, POS), ... ], ... ]
                    parsed_mor = parse_fixed( hangul_unicode_correction( parsed=self.tagger.parse(phrase) ) , join=join)  # 2-D (flattened) result: [ (morpheme, POS), ...]

                    pos_result = list() # list for the final result: 3-D (unflattened) list
                    cnt = 0 # index for a morpheme

                    for i in range(len(phrase2ej)):
                        ej_mor = list() # list for an 2-D (flattened) eojeol list: [(morpheme, POS), ...]

                        while i == int( result_mor_lst[cnt].split(",")[-1]):  # While the index of a morpheme is equal to the index of an eojeol
                            if cnt > len(parsed_mor)-1:
                                break

                            ej_mor.append(parsed_mor[cnt])  # adding a (morpheme, POS) to ej_mor
                            cnt += 1

                            if cnt == len(result_mor_lst): # If cnt is equal to the length of a phrase (or sentence)
                                break

                        pos_result.append(ej_mor) # adding the 2-D (flattened) list of an eojeol to the final 3-D (unflattened) list


                    if join == False:
                        if coda_normalization == False:
                            pos_result = [[(mor_pos[0], mor_pos[1]) for mor_pos in word] for word in pos_result]
                        elif coda_normalization == True:
                            pos_result = [ [( replace_multiple(string=mor_pos[0], replace_list=[("ᆫ", "ㄴ"), ("ᆯ", "ㄹ"), ("ᄆ", "ㅁ"), ("ᄇ", "ㅂ"), ("ᆼ", "ㅇ")]), mor_pos[1]) for mor_pos in word] for word in pos_result]

                    elif join == True:
                        if coda_normalization == False:
                            pos_result = [[mor_pos for mor_pos in word] for word in pos_result]
                        elif coda_normalization == True:
                            pos_result = [ [ replace_multiple(string=mor_pos, replace_list=[("ᆫ", "ㄴ"), ("ᆯ", "ㄹ"), ("ᄆ", "ㅁ"), ("ᄇ", "ㅂ"), ("ᆼ", "ㅇ")]) for mor_pos in word] for word in pos_result]


                    return pos_result

                    # example of pos_result
                    # [[('이것', 'NP'), ('이', 'JKS')],
                    # [('뭐', 'NP'), ('이', 'VCP'), ('ᆫ지', 'EC')],
                    # [('알', 'VV'), ('아', 'EF'), ('.', 'SF')]]


        #     else: # There is no code for Python 2. I strongly recommend you to use Python 3.
        #         phrase = phrase.encode('utf-8')
        #         if flatten:
        #             result = self.tagger.parse(phrase).decode('utf-8')
        #             return parse_fixed(result, join=join)
        #         else:
        #             return [parse_fixed(self.tagger.parse(eojeol).decode('utf-8'), join=join)
        #                     for eojeol in phrase.split()]


        else:   # If we use the original version
            """POS tagger.

            :param flatten: If False, preserves eojeols.
            :param join: If True, returns joined sets of morph and tag.
            """

            phrase = phrase.replace('영치기 영차', '영치기영차')   # a temporary solution for '영치기 영차'. '영치기 영차' consists of 2 eojeols. However, MeCab-ko analyses it as 1 eojeol. I haven't figured out the reason yet.

            if sys.version_info[0] < 3:
                phrase = phrase.encode('utf-8')
                if flatten:
                    result = self.tagger.parse(phrase).decode('utf-8')
                    return parse(result, join=join)
                else:
                    return [parse(self.tagger.parse(eojeol).decode('utf-8'), join=join)
                            for eojeol in phrase.split()]
            else:
                if flatten:
                    result = self.tagger.parse(phrase)
                    return parse(result, join=join)
                else:
                    # return [parse(self.tagger.parse(eojeol), join=join)
                    #         for eojeol in phrase.split()]

                    # flatten fixed 2021-09-26

                    ## 1) analysed result of Mecab-ko
                    result = self.tagger.parse(phrase)
                    result_mor_lst = result.splitlines()[:-1]
                    # example of result_mor_lst'
                    # ['너\tNP,*,F,너,*,*,*,*',
                    #  '를\tJKO,*,T,를,*,*,*,*',
                    #  '좋아해\tVV+EF,*,F,좋아해,Inflect,VV,EF,좋아하/VV/*+아/EF/*',
                    #  '.\tSF,*,*,*,*,*,*,*']


                    ## 2) adding indices of eojeols to result_mor_lst
                    phrase2ej = phrase.split()  # eojeol list # ['너를', '좋아해.']
                    cnt = 0  # index of an eojeol
                    concat_mor = ""

                    for i in range(len(result_mor_lst)):
                        info_str = result_mor_lst[i].split("\t")[0].strip() # '너\tNP,*,F,너,*,*,*,*'   >   '너'

                        concat_mor += info_str  # concatenating morphemes until the string is equal to their original eojeol (e.g. 알 > 알+았 > 알+았+어요)

                        if concat_mor == phrase2ej[cnt]:  # If the string of concatenated morphemes is equal to its original eojeol
                            result_mor_lst[i] += "," + str(cnt)  # adding the index (cnt) of the eojeol
                            cnt += 1
                            concat_mor = ""
                        else:
                            result_mor_lst[i] += "," + str(cnt)  # adding the index (cnt) of the eojeol
                    # example of 'result_mor_lst'
                    # ['너\tNP,*,F,너,*,*,*,*,0',   # add an eojeol index (',0') to each morpheme analysis
                    # '를\tJKO,*,T,를,*,*,*,*,0',
                    # '좋아해\tVV+EF,*,F,좋아해,Inflect,VV,EF,좋아하/VV/*+아/EF/*,1',
                    # '.\tSF,*,*,*,*,*,*,*,1']


                    ## 3) saving the 3-D (unflattened) result:    [ [ (morpheme, POS), (morpheme, POS), ... ], ... ]
                    parsed_mor = parse(self.tagger.parse(phrase), join=join)  # 2-D (flattened) result: [ (morpheme, POS), ...]

                    pos_result = list() # list for the final result: 3-D (unflattened) list
                    cnt = 0  # index for a morpheme

                    for i in range(len(phrase2ej)):
                        ej_mor = list() # list for an 2-D (flattened) eojeol list: [(morpheme, POS), ...]

                        while i == int(result_mor_lst[cnt].split(",")[-1]):  # While the index of a morpheme is equal to the index of an eojeol
                            if cnt > len(parsed_mor)-1:
                                break

                            ej_mor.append(parsed_mor[cnt])  # adding a (morpheme, POS) to ej_mor
                            cnt += 1

                            if cnt == len(result_mor_lst):  # If cnt is equal to the length of a phrase (or sentence)
                                break

                        pos_result.append(ej_mor) # adding the 2-D (flattened) list of an eojeol to the final 3-D (unflattened) list

                    return pos_result
                    # example of pos_result
                    # [[('너', 'NP'), ('를', 'JKO')], [('좋아해', 'VV+EF'), ('.', 'SF')]]


    def morphs(self, phrase):
        """Parse phrase to morphemes."""

        return [s for s, t in self.pos(phrase)]

    def nouns(self, phrase):
        """Noun extractor."""

        tagged = self.pos(phrase)
        return [s for s, t in tagged if t.startswith('N')]

    def __init__(self, dicpath='/usr/local/lib/mecab/dic/mecab-ko-dic', use_original=False):
        self.use_original = use_original    # whether to use the original version

        self.dicpath = dicpath
        try:
            self.tagger = Tagger('-d %s' % dicpath)
            self.tagset = utils.read_json('%s/data/tagset/mecab.json' % utils.installpath)
        except RuntimeError:
            raise Exception('The MeCab dictionary does not exist at "%s". Is the dictionary correctly installed?\nYou can also try entering the dictionary path when initializing the Mecab class: "Mecab(\'/some/dic/path\')"' % dicpath)
        except NameError:
            raise Exception('Install MeCab in order to use it: http://konlpy.org/en/latest/install/')

    def __setstate__(self, state):
        """just reinitialize."""

        self.__init__(dicpath=state['dicpath'])

    def __getstate__(self):
        """store arguments."""

        return {'dicpath': self.dicpath}
