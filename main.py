import sys
import os
import json
import openai
import tiktoken
from dotenv import load_dotenv
from datetime import datetime
from colorama import init
from termcolor import cprint

init()
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print('OpenAI API key not defined in .env.')
    quit()

MODEL = os.getenv('MODEL', default="gpt-3.5-turbo")
MAX_TOKENS = int(os.getenv('MAX_TOKENS', default='4000'))
DEBUG = os.getenv('DEBUG', default='false') == 'true'
ANSWER_TEMPLATE = '{"answer": "<YOUR_ANSWER>", "topic": "<TOPIC>", "follow_up_questions": ["<follow_up_question_1>", "<follow_up_question_2>", "<follow_up_question_3>", "<follow_up_question_4>", "<follow_up_question_5>", "<follow_up_question_6>"]}'

openai.api_key = OPENAI_API_KEY

print_green = lambda text: cprint(text, "green")
print_magenta = lambda text: cprint(text, "magenta")
print_cyan = lambda text: cprint(text, "cyan")
print_yellow = lambda text: cprint(text, "yellow")


def debug_log(text):
    if DEBUG:
        print_yellow(text)


def save_conversation_history(conversation):
    history_dir = "history"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_conversation.txt"
    file_path = os.path.join(history_dir, filename)

    with open(file_path, "w") as f:
        for role, content in conversation:
            f.write(f"{role.capitalize()}: {content}\n")


def count_tokens(messages, model=MODEL):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        return count_tokens(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return count_tokens(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def trim_conversation(prompt, max_tokens=MAX_TOKENS):
    while count_tokens(prompt) > max_tokens:
        if len(prompt) <= 1:
            break
        prompt.pop(1)  # Remove the second message in the prompt (first non-system message)
    return prompt


def do_request(prompt):
    prompt = trim_conversation(prompt)

    print_magenta("\nThinking...")
    try:
        response = openai.ChatCompletion.create(
            messages=prompt,
            model=MODEL,
            max_tokens=600,
            n=1,
            stop=None,
            temperature=0.5,
        )
    except openai.error.OpenAIError as e:
        raise Exception(f"OpenAI API request failed: {str(e)}")
    else:
        if response.choices and len(response.choices) > 0 and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            raise Exception("Invalid response from OpenAI API")


def build_prompt(conversation, topic=None):
    topic_formatted = f'"{topic}"' if topic else ''

    parts = []
    directives = []
    directives.append(
        'You are a research assistant that answers questions factually correct, the answers should lead to curiosity. Elaborate on the answers as much as possible. Here are your directives:')
    directives.append('1. Answer the user question.')
    if topic:
        directives.append(f'2. Come up with a topic that suits the latest 3 user questions, the topic should be short and descriptive.')
    directives.append(
        f'{"3" if topic else "2"}. Create {"6" if topic else "5"} follow-up questions for your current answer. The questions should be directed to the assistant.{f" Try and stay as close to the topic {topic_formatted} as possible, unless the user clearly changes interest. The 6th follow-up question should derive from the topic a bit, while still being related to the last user message. " if topic else ""}.')
    parts.append('\n'.join(directives))
    if len(conversation):
        parts.append(
            'Decrease repetition of answers and follow-up questions by analyzing the chat history. Also, stay away from repetitive follow-up questions in general.')

    parts.append(
        f'IMPORTANT: Only respond in JSON formatting using the following template exactly:\n\n{ANSWER_TEMPLATE}')
    system_content = '\n\n'.join(parts)

    prompt = [{"role": "system", "content": system_content}]
    for msg in conversation:
        role, content = msg
        prompt.append({"role": role, "content": content})

    debug_log('\nPrompt: {}'.format(json.dumps(prompt, indent=4)))

    return prompt


def parse_response(response):
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse response as JSON: {str(e)}")

    if data:
        answer = data.get("answer", "").strip()
        topic = data.get("topic", "").strip()
        options = data.get("follow_up_questions", [])
        return answer, topic, options
    else:
        raise Exception("Invalid response from OpenAI API")


def ask(question, conversation, topic=None):
    conversation.append(("user", question))

    prompt = build_prompt(conversation, topic)
    response = do_request(prompt)

    try:
        answer, topic, options = parse_response(response)
    except Exception as e:
        print(f"Error: {str(e)}")
        return False, None, None

    debug_log(f'\nCurrent topic: {topic if topic else "None"}')

    if answer:
        print_green(f"\n{answer}")
        success = True

    if options:
        options = list(map(lambda x: x.strip(), options))

    if DEBUG and not answer or not topic or not options:
        print_yellow(f'Could not parse response answer, topic or options from: {response}')

    return success, answer, topic, options


def ask_and_append(conversation, question, topic=None):
    success, answer, updated_topic, options = ask(question, conversation, topic)
    topic = updated_topic if updated_topic else topic

    if success:
        conversation.append(("assistant", answer))
        save_conversation_history(conversation)

    return success, answer, topic, options


def main():
    conversation = []
    topic = None

    while True:
        if not len(conversation):
            question = input("\nEnter a question or type 'exit' to quit:\n")
        else:
            if options:
                print('')
                for i, option in enumerate(options, 1):
                    print_cyan(f"{i}. {option}")

                user_input = input(
                    '\nChoose a follow-up question by entering the corresponding number (1-5) or type a custom question (type exit to quit):\n')

                if user_input.isdigit() and 1 <= int(user_input) <= len(options):
                    question = options[int(user_input) - 1]
                else:
                    question = user_input
            else:
                question = input('\nAsk a follow-up question:\n')

        if question.lower() == "exit":
            print('\n\nBye-bye!')
            return

        success, answer, topic, options = ask_and_append(conversation, question, topic)

        if not success:
            print("\nSorry, I couldn't find an answer to your question. Try another one.\n")
            continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nBye-bye!')
        sys.exit()
