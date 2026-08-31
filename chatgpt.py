import os

from openai import OpenAI

client = OpenAI()


def generate_label(description):
    prompt = """
    Use the following bullet points to respond to user inputs.
    * User will provide you complaints between parties as their discussions.
    * Act as an experienced legal agent who specializes in handling a wide range of disputes, including digital products, subscriptions, physical products, and so on. You have a deep understanding of legal nuances, emotional undertones, and the potential impacts of each case. Meaning- "You are an agent with real life expertise as consumer complaints resolver just like a member in Local Better Business Bureaus, Keep this in mind."
    * Analyze the detailed discussion  and categorize the severity of the dispute as High, Medium, or Low based on the financial stakes, legal ramifications, emotional distress, and overall urgency evident in the dialogue.
    Note: Only respond with one word: High, Medium, or Low.
    """
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',  # Specify the model here
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": description}
        ],
        max_tokens=10,  # Enough tokens to get a short classification response
        n=1,
        stop=None,
        temperature=0
    )
    return response.choices[0].message.content


label = generate_label("This Fake Girl Profile Banned More times and Scammer. This scammer making a women profiles but all the prices a very cheap and previous banned profiles a the same look @Hulk @Middleman Please take an action")

print(label)
