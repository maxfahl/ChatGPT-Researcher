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

answer_template = '{"topic": "<TOPIC>", "answer": "<YOUR_ANSWER>", "follow_up_questions": ["<follow_up_question_1>", "<follow_up_question_2>", "<follow_up_question_3>", "<follow_up_question_4>", "<follow_up_question_5>"]}'

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


def get_num_tokens(text):
    return len(encoder.encode(text))


def accept_token_length(text):
    num_tokens = get_num_tokens(text)
    return num_tokens <= max_request_tokens


def trim_qas():
    qas_formatted = get_qas_formatted()
    while not accept_token_length(qas_formatted):
        previous_qas.pop()
        qas_formatted = get_qas_formatted()


def ask(question, topic=None):
    has_history = len(previous_qas) != 0
    parts = []
    parts.append('--- START OF CONTEXTUAL INFORMATION')
    if has_history:
        parts.append(f'[Q&A HISTORY]\n{get_qas_formatted()}')
    if topic:
        parts.append(f'[TOPIC]\n{topic}')
    parts.append(f'[CURRENT QUESTION]\n{question}')
    parts.append('--- END OF CONTEXTUAL INFORMATION')
    parts.append('--- START OF PROMPT')
    directives = []
    directives.append(
        'You are an AI researcher that answers questions factually correct, the answers should lead to curiosity. Elaborate on the answers as much as possible. Here are your directives:')
    directives.append('1. Answer the [CURRENT QUESTION].')
    if topic:
        parts.append(f'2. Come up with a topic that suits the latest 2 questions of [Q&A HISTORY], the topic should be short but descriptive.')
    directives.append(
        f'{"3" if topic else "2"}. Create 5 follow-up questions for your current answer.{" Try and stay as close to the [TOPIC] as possible." if topic else ""}')
    parts.append('\n'.join(directives))
    if len(previous_qas):
        parts.append(
            'Analyze the [Q&A HISTORY] to decrease repetition of answers and follow-up questions.')
    parts.append(
        'Try to keep the answers relevant for the [TOPIC] as possible, and do not diverge from it as long as the user\'s questions does not indicate a clear change of interest.')
    parts.append(
        f'IMPORTANT: Only respond in JSON formatting following the following template exactly:\n\n{answer_template}')
    prompt = '\n\n'.join(parts)

    response = do_request(prompt)

    success = False
    data = None
    answer = None
    options = None
    topic = None
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        pass

    if data:
        answer = data["answer"].strip() if data["answer"] else None
        topic = data["topic"].strip() if data["topic"] else None
        options = data["follow_up_questions"] if data["follow_up_questions"] else None

    if answer:
        print_green(f"\n{answer}")
        previous_qas.append(f'[QUESTION]: {question}\n[ANSWER]: {answer}')
        trim_qas()
        success = True
    # else:
    #     print(response)

    # if topic:
    #     print(f'Current topic: {topic}')

    if options:
        options = list(map(lambda x: x.strip(), options))

    return success, options, topic


def main():
    while True:
        original_question = input("\nEnter a question:\n")
        success, options, topic = ask(original_question)
        topic = topic if topic else original_question

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

            success, options, updated_topic = ask(follow_up_question, topic)
            topic = updated_topic if updated_topic else topic


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nBye-bye!')
        quit()
