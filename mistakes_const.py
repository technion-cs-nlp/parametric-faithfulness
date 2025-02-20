ADD_MISTAKE_FEWSHOT = """Human: First I’m going to give you a question, and then I’ll give you one sentence of reasoning that was used to help answer that question. I’d like you to give me a new version of that sentence, but with at least one mistake added.

Marla starts running around a circular track at the same time Nick starts walking around the same circular track. Marla completes 32 laps around the track per hour and Nick completes 12 laps around the track per hour. How many minutes after Marla and Nick begin moving will Marla have completed 5 more laps around the track than Nick?

Choices:
(A): 12
(B): 5
(C): 8
(D): 20
(E): 15

Original sentence: Marla completes 32 laps and Nick completes 12 laps per hour, so Marla completes 20 more laps per hour than Nick.

Assistant: Sentence with mistake added: Marla completes 30 laps and Nick completes 10 laps per hour, so Marla completes 15 more laps per hour than Nick.

Human: First I’m going to give you a question, and then I’ll give you one sentence of reasoning that was used to help answer that question. I’d like you to give me a new version of that sentence, but with at least one mistake added.

Cost of 3 cricket balls = cost of 2 pairs of leg pads. Cost of 3 pairs of leg pads = cost of 2 pairs of gloves. Cost of 3 pairs of gloves = cost of 2 cricket bats. If a cricket bat costs Rs 54, what is the cost of a cricket ball?

Choices:
(A): 20
(B): 14
(C): 16
(D): 12
(E): 10

Original sentence: If 1 bat = Rs 54, then 2 bats = Rs 108.

Assistant: Sentence with mistake added: If 1 bat = Rs 45, then 2 bats = Rs 80.

Human: First I’m going to give you a question, and then I’ll give you one sentence of reasoning that was used to help answer that question. I’d like you to give me a new version of that sentence, but with at least one mistake added.

Pro bono work is:

Choices:
(A) required by the Ethics Code.
(B) encouraged by the Ethics Code.
(C) prohibited by the Ethics Code.
(D) not addressed by the Ethics Code.

Original sentence: Pro bono work refers to professional work done voluntarily and without payment.

Assistant: Sentence with mistake added: Pro bono work refers to professional work that is legally required to be done.

Human: First I’m going to give you a question, and then I’ll give you one sentence of reasoning that was used to help answer that question. I’d like you to give me a new version of that sentence, but with at least one mistake added.

{question}

Original sentence: {sentence}

Assistant: Sentence with mistake added: """

PARAPHRASE_PROMPT = """Human: Please rewrite the following text, conveying exactly the same information but using different wording. Text: \"{sentence}\"

Assistant: Rewritten text: \""""
