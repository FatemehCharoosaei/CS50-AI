from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # A can be a knight or a knave but can't be both:
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    # If A be a knight her sentence is True:
    Implication(AKnight, And(AKnight, AKnave)),
    # If A be a knave her sentence is False:
    Implication(AKnave, Not(And(AKnight, AKnave))),
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # A and B can be knights or knaves but can't be both:
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    # If A be a knight, A and B are both knaves:
    Implication(AKnight, And(AKnave, BKnave)),
    # If A be a knave, her statement is False:
    Implication(AKnave, Not(And(AKnave, BKnave))),
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # A and B can be knights or knaves but can't be both:
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    # If A be a knight, A and B are both the same:
    Implication(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    # If A be a knave, her statement is False:
    Implication(AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave)))),
    # If B be a knight, A and B are not the same:
    Implication(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),
    # If B be a knave, her statement is False:
    Implication(BKnave, Not(Or(And(AKnight, BKnave), And(AKnave, BKnight)))),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # A, B and C can be knights or knaves but can't be both:
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    And(Or(CKnight, CKnave), Not(And(CKnight, CKnave))),
    # If B be a knight, A said 'I am a knave', and C is a knave:
    Implication(BKnight, CKnave),
    Implication(
        BKnight,
        And(
            # A then said 'I am a Knave', A may be a Knight or a Knave:
            Implication(AKnight, AKnave),
            Implication(AKnave, Not(AKnave)),
        ),
    ),
    # If B be a knave, A said 'I am a knight' C is not a knave:
    Implication(BKnave, Not(CKnave)),
    Implication(
        BKnave,
        And(
            # A then said 'I am a Knight', A may be a Knight or a Knave:
            Implication(AKnight, AKnight),
            Implication(AKnave, Not(AKnight)),
        ),
    ),
    # If C be a knight, A is a knight:
    Implication(CKnight, AKnight),
    # If C be a knave, A is not a knight:
    Implication(CKnave, Not(AKnight)),
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3),
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
