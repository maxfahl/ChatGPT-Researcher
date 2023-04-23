from decouple import config
import openai
import re
import json
import tiktoken
from colorama import init
from termcolor import colored, cprint

encoder = tiktoken.encoding_for_model('gpt-3.5-turbo')

init()

OPENAI_API_KEY = config('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print('OpenAI API key not defined in .env.')
    quit()

openai.api_key = OPENAI_API_KEY
max_request_tokens = 2000

answer_template = '{"topic": <TOPIC>, "answer": "<YOUR_ANSWER>", "follow_up_questions": ["<follow_up_question_1>", "<follow_up_question_2>", "<follow_up_question_3>", "<follow_up_question_4>", "<follow_up_question_5>"]}'

print_green = lambda text: cprint(text, "green")
print_blue = lambda text: cprint(text, "blue")
print_magenta = lambda text: cprint(text, "magenta")
print_cyan = lambda text: cprint(text, "cyan")

messages = [{
    "role": "system",
    "content": f"""You are an AI researcher that answers questions factually correct, the answers should lead to curiosity. Elaborate on the answers as much as possible. Here are your directives:
    
    1. Answer the question submitted by the user.
    2. Create 5 follow-up questions for your current answer, but also analyze the previous questions to determine the follow-up questions based on what seems to interest the user.
    
    Analyze the previous questions and answers to decrease repetition of answers and follow-up questions.
    
    The format of your answer should be in JSON following the [ANSWER TEMPLATE] exactly.
    
    [ANSWER TEMPLATE]
    {answer_template}"""
}]


# def num_tokens_from_messages():
#     num_tokens = len(encoder.encode(text))
#     return num_tokens <= max_request_tokens


# def do_request(question):
#     print_magenta("\nThinking...")
#     messages.append({"role": "user", "content": question})
#     response = openai.ChatCompletion.create(
#         messages=messages,
#         model="gpt-3.5-turbo",
#         max_tokens=600,
#         n=1,
#         stop=None,
#         temperature=0.25,
#     )
#     return response.choices[0].message.content.strip()


def accept_token_length(text):
    num_tokens = len(encoder.encode(text))
    return num_tokens <= max_request_tokens


def ask(question):
    print_magenta("\nThinking...")

    messages.append({"role": "user", "content": question})
    response = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=600,
        n=1,
        stop=None,
        temperature=0.25,
    )
    response = response.choices[0].message.content.strip()
    # response = do_request(question)

    success = False
    data = None
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        pass

    print(data)

    answer = data["answer"].strip() if data and data["answer"] else None
    options = data["follow_up_questions"] if data and data["follow_up_questions"] else None

    if answer:
        print_green(f"\n{answer}")
        messages.append({"role": "assistant", "content": answer})
        # trim_qas()
        success = True

    # print(f'Num tokens: {num_tokens_from_messages()}')
    return success, options


def main():
    while True:
        original_question = input("\nEnter a question:\n")
        success, options = ask(original_question)

        while True:
            if not success:
                print("\nNo answer found. Try another question.\n")
                break

            if options:
                print('')
                for i, option in enumerate(options, 1):
                    print_cyan(f"{i}. {option}")

                user_input = input(
                    '\nChoose a follow-up question by entering the corresponding number (1-5) or type a custom question:\n')

                if user_input.isdigit() and 1 <= int(user_input) <= 5:
                    follow_up_question = options[int(user_input) - 1]
                else:
                    follow_up_question = user_input
            else:
                follow_up_question = input('\nCould not find any follow-up questions, please provide one yourself:\n')

            success, options = ask(follow_up_question)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nBye-bye!')
        quit()
