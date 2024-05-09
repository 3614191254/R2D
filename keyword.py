import torch
import nltk
nltk.download('stopwords')
from transformers import BertTokenizer, BertForMaskedLM
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize


# 加载BERT模型和Tokenizer
model_name = "bert-base-uncased"
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 加载停用词
stop_words = set(stopwords.words("english"))

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 读取文本文件
file_path = "input.txt"
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

        # 转换为PyTorch张量并传递到设备上
        input_ids_tensor = torch.tensor([input_ids]).to(device)

        # 使用BERT进行预测
        with torch.no_grad():
            outputs = model(input_ids_tensor)

        # 获取预测结果中的概率最大的前100个单词
        _, predicted_indexes = torch.topk(outputs.logits[0, i], 100)
        predicted_words = tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())

        # 判断是否找到关键词且不是停用词
        if word not in predicted_words and word not in stop_words:
            keywords.append(word)
    
    return keywords
def filter_nouns_and_verbs(words):
    filtered_words = []
    for word in words:
        # 使用NLTK的词性标注来确定单词的词性
        pos_tags = pos_tag(word_tokenize(word))
        # 只保留动词（VB）和名词（NN）
        if any(tag.startswith('VB') or tag.startswith('NN') for _, tag in pos_tags):
            filtered_words.append(word)
    return filtered_words
# 对每个句子生成关键词，并将结果写入output.txt文件
with open("keyword.txt", "w", encoding="utf-8") as output_file:
    for i, sentence in enumerate(sentences):
        result_keywords1 = generate_keywords(sentence)
        result_keywords = filter_nouns_and_verbs(result_keywords1)
        output_file.write(f"{result_keywords}\n")