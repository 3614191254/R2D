import openai
import time

openai.api_key = 'sk-yours-key'

def ask_question(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Q:You are a developer.When someone have the problem of Auto-Backup failing, Which line of the following text describes the most relevant content to the problem of Backup failing？A:It is relevant to the description:Provides a comprehensive framework for user interface customization, navigation, synchronization, and handling integrated Trakt actions within the SeriesGuide app. It is the 89th line of the text. They are relevant because they directly involve synchronization and maintenance functions in the application, which is part of the comprehensive framework provided by the application. "},
            {"role": "user", "content": prompt}
        ]
    )
    best_row = response.choices[0].message.content
    return best_row

def main():
    with open('your file path', 'r', encoding='utf-8') as file:
        a_content = file.read()
    a = a_content.split(',')
    with open('your file path', 'r', encoding='utf-8') as file:
        b_content = file.readlines()

    # 打开结果文件
    with open('your file path', 'w', encoding='utf-8') as result_file:
        for i, string in enumerate(a):
            prompt = f" Which line of the following text describes the most relevant content to '{string.strip()}'? Please only answer the number of lines in the text where this sentence is located. \n Let's work this out in a step by step way to be sure we have the right answer.The text content is as follows:\n{b_content}"
            best_row = ask_question(prompt)
            result_file.write(f"{best_row}\n")
            time.sleep(5)

if __name__ == "__main__":
    main()
