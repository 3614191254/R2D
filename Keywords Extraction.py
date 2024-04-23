import openai
import torch
import nltk
nltk.download('stopwords')
from transformers import BertTokenizer, BertForMaskedLM
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize


openai.api_key = 'sk-yours-key'
model_name = "bert-base-uncased"
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 加载停用词
stop_words = set(stopwords.words("english"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

file_path = "data-path"
with open(file_path, "r", encoding="utf-8") as file:
    sentences = file.readlines()

# 逐个句子进行关键词提取和停用词过滤
def generate_keywords(sentence):
    keywords = []

    words = tokenizer.tokenize(sentence)
    
    for i, word in enumerate(words):
        # 创建一个带有 [MASK] 的输入
        input_ids = tokenizer.convert_tokens_to_ids(words)
        input_ids[i] = tokenizer.mask_token_id
        input_ids_tensor = torch.tensor([input_ids]).to(device)
        with torch.no_grad():
            outputs = model(input_ids_tensor)
        _, predicted_indexes = torch.topk(outputs.logits[0, i], 100)
        predicted_words = tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
        if word not in predicted_words and word not in stop_words:
            keywords.append(word)
    
    return keywords

def filter_nouns_and_verbs(words):
    filtered_words = []
    for word in words:
        # 使用NLTK的词性标注来确定单词的词性
        pos_tags = pos_tag(word_tokenize(word))
        if any(tag.startswith('VB') or tag.startswith('NN') for _, tag in pos_tags):
            filtered_words.append(word)
    return filtered_words

def generate_keyword_phrases(keywords):
    phrases = []
    for i in range(len(keywords)):
        if i < len(keywords) - 1 and keywords[i+1] in tokenizer.vocab:
            phrases.append(keywords[i] + " " + keywords[i+1])
        elif i > 0 and i < len(keywords) - 1:
            phrase_candidate = keywords[i-1] + " " + keywords[i]
            pos_tags = pos_tag(word_tokenize(phrase_candidate))
            if any(tag.startswith('VB') or tag.startswith('NN') for _, tag in pos_tags):
                phrases.append(phrase_candidate)
    return phrases

def filter_illogical_phrases(phrases):
    filtered_phrases = []
    for phrase in phrases:
        
        response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Does this phrase align with logic?\"{phrase}\"\n yes or no:"}
        ]
        
        )
        # 判断逻辑是否合理
        if response.choices[0].message.content == "yes":
            filtered_phrases.append(phrase)
    return filtered_phrases

# 对每个句子生成关键词，并将结果写入文件
with open("your-file-path", "w", encoding="utf-8") as output_file:
    for i, sentence in enumerate(sentences):
        result_keywords = generate_keywords(sentence)
        if result_keywords:
            result_phrases = generate_keyword_phrases(result_keywords)
            if result_phrases:
                filtered_phrases = filter_illogical_phrases(result_phrases)
                if filtered_phrases:
                    output_file.write(f"{filtered_phrases}\n")
                else:
                    output_file.write(f"\n")
            else:
                filtered_words = filter_nouns_and_verbs(result_keywords)
                output_file.write(f"{filtered_words}\n")
        else:
            output_file.write(f"\n")
