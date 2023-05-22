from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# Set up the OpenAI API credentials
openai.api_key = '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'

@app.route('/', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data['query']

    # Generate a response using ChatGPT
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=query,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )

    return jsonify({'response': response.choices[0].text.strip()})

if __name__ == '__main__':
    app.run()
