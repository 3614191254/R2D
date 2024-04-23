import openai


openai.api_key = 'sk-yours-key'


def sentiment_analysis(text):
    b = f"This is a text sentiment analysis task.Pleaase answer the Sentiment(negative or positive):\n"
    a = b + f"{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": a}
        ]
        
    )
    sentiment = response.choices[0].message.content
    if sentiment == "Positive":
        return "positive"
    elif sentiment == "Negative":
        return "negative"
    else:
        return "unknown"  # 情感分析结果无法确定

def main():
    with open('your file path', 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    for i, sentence in enumerate(sentences):
        sentiment = sentiment_analysis(sentence)
        if sentiment == "positive":
            with open('your file path', 'a', encoding='utf-8') as result_file:
                result_file.write(f"{sentence.strip()}\n")
        elif sentiment == "negative":
            with open('your file path', 'a', encoding='utf-8') as result_file:
                result_file.write(f"{sentence.strip()}\n")
        else:
            print(f"Unable to determine sentiment for sentence {i + 1}: {sentence.strip()}")

if __name__ == "__main__":
    main()
