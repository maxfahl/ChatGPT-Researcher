from decouple import config
import openai
import re
import json
import tiktoken
from colorama import init
from termcolor import colored, cprint


init()

OPENAI_API_KEY = config('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print('OpenAI API key not defined in .env.')
    quit()
openai.api_key = OPENAI_API_KEY

DEBUG = config('DEBUG') == 'true'
ANSWER_TEMPLATE = '{"answer": "<YOUR_ANSWER>", "topic": "<TOPIC>", "follow_up_questions": ["<follow_up_question_1>", "<follow_up_question_2>", "<follow_up_question_3>", "<follow_up_question_4>", "<follow_up_question_5>", "<follow_up_question_6>"]}'
MAX_HISTORY_TOKENS = 3000

encoder = tiktoken.encoding_for_model('gpt-3.5-turbo')
qa_history = []

print_green = lambda text: cprint(text, "green")
print_blue = lambda text: cprint(text, "blue")
print_magenta = lambda text: cprint(text, "magenta")
print_cyan = lambda text: cprint(text, "cyan")
print_yellow = lambda text: cprint(text, "yellow")


def debug_log(text):
    if DEBUG:
        print_yellow(text)


def get_num_tokens(text):
    return len(encoder.encode(text))


def has_qa_history():
    return len(qa_history) > 0


def get_qa_history_formatted():
    return '\n'.join(qa_history)


def previous_qa_history_length_ok():
    qas_formatted = get_qa_history_formatted()
    num_tokens = get_num_tokens(qas_formatted)
    return num_tokens <= MAX_HISTORY_TOKENS


def trim_qas():
    while not previous_qa_history_length_ok():
        qa_history.pop()
        trim_qas()


def do_request(prompt):
    debug_log(f'Prompt:\n\n{prompt}');
    print_magenta("\nThinking...")
    response = openai.ChatCompletion.create(
        messages=[{"role": "system", "content": prompt}],
        model="gpt-3.5-turbo",
        max_tokens=600,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def ask(question, topic=None):
    parts = []
    parts.append('--- START OF CONTEXT')
    if has_qa_history():
        parts.append(f'[Q&A HISTORY]\n{get_qa_history_formatted()}')
    if topic:
        parts.append(f'[TOPIC]\n{topic}')
    parts.append(f'[CURRENT QUESTION]\n{question}')
    parts.append('--- END OF CONTEXT')
    parts.append('--- START OF PROMPT')
    directives = []
    directives.append(
        'You are an AI researcher that answers questions factually correct, the answers should lead to curiosity. Elaborate on the answers as much as possible. Here are your directives:')
    directives.append('1. Answer the [CURRENT QUESTION].')
    if topic:
        directives.append(f'2. Come up with a topic that suits the latest 3 questions of [Q&A HISTORY], the topic should be short but descriptive.')
    directives.append(
        f'{"3" if topic else "2"}. Create 5 follow-up questions for your current answer.{" Try and stay as close to the [TOPIC] as possible." if topic else ""}. The 6th follow-up question should derive from the [TOPIC], by still being related to the [CURRENT QUESTION].')
    parts.append('\n'.join(directives))
    if len(qa_history):
        parts.append(
            'Analyze the [Q&A HISTORY] to decrease repetition of answers and follow-up questions.')

    parts.append(
        f'IMPORTANT: Only respond in JSON formatting using the following template exactly:\n\n{ANSWER_TEMPLATE}')
    prompt = '\n\n'.join(parts)

    response = do_request(prompt)

    success = False
    data = None
    answer = None
    topic = None
    options = None

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
        qa_history.append(f'Question: {question}\nAnswer: {answer}')
        trim_qas()
        success = True

    if options:
        options = list(map(lambda x: x.strip(), options))

    debug_log(f'Topic: {topic if topic else "None"}')

    if DEBUG and not answer or not topic or not options:
        print_yellow(f'Could not parse response answer, topic or options from: {response}')

    return success, topic, options


def main():
    while True:
        original_question = input("\nEnter a question:\n")
        success, topic, options = ask(original_question)
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

                if user_input.isdigit() and 1 <= int(user_input) <= 6:
                    follow_up_question = options[int(user_input) - 1]
                else:
                    follow_up_question = user_input
            else:
                follow_up_question = input('\nCould not find any follow-up questions, please provide one yourself:\n')

            success, updated_topic, options = ask(follow_up_question, topic)
            topic = updated_topic if updated_topic else topic


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nBye-bye!')
        quit()
