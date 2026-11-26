from medreason.rewards import (
    correctness_reward,
    extract_answer,
    make_format_reward,
    make_length_penalty,
)

GOOD = "<think>The patient has X, so Y.</think>\n<answer>B</answer>"


def test_extract_answer():
    assert extract_answer(GOOD) == "B"
    assert extract_answer("<answer> c </answer>") == "C"
    assert extract_answer("The answer is B") is None


def test_correctness_reward():
    assert correctness_reward([GOOD, GOOD], answer=["B", "A"]) == [1.0, 0.0]


def test_format_reward_requires_both_tags():
    reward = make_format_reward(0.2)
    assert reward([GOOD]) == [0.2]
    assert reward(["<answer>B</answer>"]) == [0.0]
    assert reward([GOOD + " extra text"]) == [0.0]


def test_format_reward_rejects_empty_reasoning():
    assert make_format_reward(0.2)(["<think></think><answer>B</answer>"]) == [0.0]


def test_length_penalty():
    penalty = make_length_penalty(budget_words=3, penalty_per_word=0.1)
    assert penalty(["a b c"]) == [0.0]
    assert abs(penalty(["a b c d e"])[0] + 0.2) < 1e-9
