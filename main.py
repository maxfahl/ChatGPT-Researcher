from decouple import config
import openai
import re
import tiktoken
from colorama import init
from termcolor import colored, cprint

# import subprocess
# import sys
#
# # Add the path to the installed packages to the Python path
# sys.path.append('.')
#
# # Read the requirements.txt file
# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()
#
# # Print message to indicate the installation process has started
# print("Installing dependencies...")
#
# # Check if each requirement is already installed
# for requirement in requirements:
#     try:
#         __import__(requirement)
#     except ImportError:
#         # If the package is not installed, install it
#         subprocess.check_call(['pip', 'install', requirement], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#
# # Print message to indicate the installation process has finished
# print("Done!")

encoder = tiktoken.encoding_for_model('gpt-3.5-turbo')

init()

OPENAI_API_KEY = SECRET_KEY = config('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print('OpenAI API key not defined in .env.')
    quit()

openai.api_key = ""
max_request_tokens = 1000

answer_template = """<The answer>
+++
1. Follow up question 1
2. Follow up question 2
3. Follow up question 3
4. Follow up question 4
5. Follow up question 5"""

print_green = lambda text: cprint(text, "green")
print_blue = lambda text: cprint(text, "blue")
print_magenta = lambda text: cprint(text, "magenta")
print_cyan = lambda text: cprint(text, "cyan")

previous_qas = []


def get_qas_formatted():
    return '\n\n'.join(previous_qas)


def do_request(prompt):
    print_magenta("\nThinking...")
    response = openai.ChatCompletion.create(
        messages=[{"role": "system", "content": prompt}],
        model="gpt-3.5-turbo",
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=1.2,
    )
    return response.choices[0].message.content.strip()


def accept_token_length(text):
    num_tokens = len(encoder.encode(text))
    return num_tokens <= max_request_tokens


def trim_qas():
    qas_formatted = get_qas_formatted()
    while not accept_token_length(qas_formatted):
        previous_qas.pop()
        qas_formatted = get_qas_formatted()


def ask(question, original_question=None):
    parts = []
    parts.append('[START OF CONTEXTUAL INFORMATION]')
    if len(previous_qas):
        parts.append(f'[Q&A HISTORY]\n{get_qas_formatted()}')
    if original_question:
        parts.append(f'[ORIGINAL QUESTION]\n{original_question}')
    parts.append(f'[CURRENT QUESTION]\n{question}')
    parts.append(f'[ANSWER TEMPLATE]\n{answer_template}')
    parts.append('[END OF CONTEXTUAL INFORMATION]')
    parts.append('[START OF PROMPT]')
    parts.append(
        'You are an AI researcher that answers questions as factually correct as possible, the answers shall inspire the user and lead to more curiosity. Elaborate as much as possible in the answer, let it be long if there are lots to tell.')
    parts.append('You main goal is to answer the question under [CURRENT QUESTION].')
    parts.append(
        'Your secondary goal is to make up 5 follow-up questions related to your answer. Make the follow-up questions as diverse as possible, and inspire curiosity. Once in a while, sprinkle in a silly or funny follow-up question. Take the [Q&A HISTORY] into consideration not to make silly or funny follow-up questions too often.')
    if len(previous_qas):
        parts.append('''Analyse the chat history under [Q&A HISTORY] to:
        a. Not repeat answers
        b. Not repeat follow-up questions
        c. To find what most interests the user, using that insight to find the best possible follow-up questions.''')
    parts.append(
        'Keep the answers close to the the question under [ORIGINAL QUESTION] as much as possible, and do not diverge as long as the user\'s questions does not indicate a clear change of interest.')
    parts.append(
        'Finally, the format of your answer should follow the template under [ANSWER TEMPLATE] exactly, and make sure that the "+++" part of the template is not removed (very important).')
    prompt = '\n\n'.join(parts)

    response = do_request(prompt)
    split = response.split("+++")
    answer = split[0].strip()
    previous_qas.append(f'[QUESTION]: {question}\n[ANSWER]: {answer}')
    trim_qas()
    print_green(f"\n{answer}")
    options_string = split[1].strip() if len(split) > 1 else None
    options = re.findall(r'\d\.\s(.*?)(?:\n|$)', options_string) if options_string else []
    return options


def main():
    while True:
        original_question = input("\nWhat are you wondering about? ")
        options = ask(original_question)

        while True:
            if len(options) == 0:
                print("\nNo follow-up questions found. Please try a different topic.\n")
                break
            else:
                print('')

            for i, option in enumerate(options, 1):
                print_cyan(f"{i}. {option}")

            user_input = input('\nChoose a follow-up question by entering the corresponding number (1-5) or type a custom question: ')

            if user_input.isdigit() and 1 <= int(user_input) <= 5:
                follow_up_question = options[int(user_input) - 1]
            else:
                follow_up_question = user_input

            options = ask(follow_up_question, original_question)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nBye-bye!')
        quit()
