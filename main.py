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

# answer_template = """<The answer>
# ++DO NOT REMOVE++
# 1. Follow up question 1
# 2. Follow up question 2
# 3. Follow up question 3
# 4. Follow up question 4
# 5. Follow up question 5"""

answer_template = '{"answer": "<YOUR_ANSWER>", "follow_up_questions": ["<follow_up_question_1>", "<follow_up_question_2>", "<follow_up_question_3>", "<follow_up_question_4>", "<follow_up_question_5>"]}'

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
        max_tokens=500,
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
    has_history = len(previous_qas) != 0
    parts = []
    parts.append('[START OF CONTEXTUAL INFORMATION]')
    if has_history:
        parts.append(f'[Q&A HISTORY]\n{get_qas_formatted()}')
    if original_question:
        parts.append(f'[ORIGINAL QUESTION]\n{original_question}')
    parts.append(f'[CURRENT QUESTION]\n{question}')
    parts.append(f'[ANSWER TEMPLATE]\n{answer_template}')
    parts.append('[END OF CONTEXTUAL INFORMATION]')
    parts.append('[START OF PROMPT]')
    parts.append(
        'You are an AI researcher that answers questions factually correct, the answers should lead to curiosity. Elaborate on the answer as much as possible.')
    parts.append('Goal 1: Answer the [CURRENT QUESTION].')
    parts.append(
        f'Goal 2: Make up 5 follow-up questions mainly related to your answer. {" Also take the previous questions under [Q&A HISTORY] into consideration to try and figure out what interests the questioner." if has_history else ""}')
    if len(previous_qas):
        parts.append(
            'When goal 1 and 2 are complete, analyse [Q&A HISTORY] to decrease repetition of answers and follow-up questions')
    parts.append(
        'Always try to keep the answers as close to the the [ORIGINAL QUESTION] as possible, and do not diverge from it as long as the user\'s questions does not indicate a clear change of interest.')
    parts.append(
        'The format of your answer should be in JSON following the [ANSWER TEMPLATE] exactly.')
    prompt = '\n\n'.join(parts)

    response = do_request(prompt)

    success = False
    data = None
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        pass

    answer = data["answer"].strip() if data["answer"] else None
    options = data["follow_up_questions"] if data["follow_up_questions"] else None

    if answer:
        print_green(f"\n{answer}")
        previous_qas.append(f'[QUESTION]: {question}\n[ANSWER]: {answer}')
        trim_qas()
        success = True

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

            success, options = ask(follow_up_question, original_question)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nBye-bye!')
        quit()
