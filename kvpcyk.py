from collections import defaultdict


def extract_rules_from_tree(tree):
    rules = []
    if len(tree) == 2 and isinstance(tree[1], str):  # Terminal rule
        lhs, word = tree
        rules.append((lhs, (word,)))
    elif len(tree) == 3:
        lhs, left, right = tree
        rules.append((lhs, (left[0], right[0])))
        rules += extract_rules_from_tree(left)
        rules += extract_rules_from_tree(right)
    return rules

def compute_rule_probabilities(trees):
    rule_counts = defaultdict(int)
    lhs_counts = defaultdict(int)

    for tree in trees:
        rules = extract_rules_from_tree(tree)
        for lhs, rhs in rules:
            rule_counts[(lhs, rhs)] += 1
            lhs_counts[lhs] += 1

    rule_probs = {
        (lhs, rhs): count / lhs_counts[lhs]
        for (lhs, rhs), count in rule_counts.items()
    }

    return rule_probs

def collect_leaves(tree):
    if len(tree) == 2 and isinstance(tree[1], str):
        return [tree[1]]
    elif len(tree) == 3:
        return collect_leaves(tree[1]) + collect_leaves(tree[2])
    return []

def validate_and_score_tree(tree, sentence, rule_probs):
    leaves = collect_leaves(tree)
    matches = (leaves == sentence)
    
    return matches

def compute_tree_probability(tree, rule_probs):
    if len(tree) == 2 and isinstance(tree[1], str):
        lhs, word = tree
        rule = (lhs, (word,))
        if rule not in rule_probs:
            raise ValueError(f"Unknown rule: {lhs} -> '{word}'")
        return rule_probs[rule]
    elif len(tree) == 3:
        lhs, left, right = tree
        rule = (lhs, (left[0], right[0]))
        if rule not in rule_probs:
            raise ValueError(f"Unknown rule: {lhs} -> {left[0]} {right[0]}")
        left_prob = compute_tree_probability(left, rule_probs)
        right_prob = compute_tree_probability(right, rule_probs)
        return rule_probs[rule] * left_prob * right_prob
    else:
        raise ValueError("Invalid tree format")

training_trees = [
    ('S', ('NP', 'she'), ('VP', ('V', 'eats'), ('NP', 'fish'))),
    ('S', ('NP', 'fish'), ('VP', 'eats')),
    ('S', ('NP', 'she'), ('VP', 'eats')),
    ('S', ('NP', 'fish'), ('VP', ('V', 'eats'), ('NP', 'fish')))
]

rule_probs = compute_rule_probabilities(training_trees)

sentence = ['she', 'eats', 'fish']
test_tree = (
    'S',
    ('NP', 'she'),
    ('VP',
        ('V', 'eats'),
        ('NP', 'fish')
    )
)

is_valid = validate_and_score_tree(test_tree, sentence, rule_probs)

print("Sentence Valid for Tree?", is_valid)
print("\nLearned Grammar Rules with Probabilities:")
for (lhs, rhs), p in rule_probs.items():
    print(f"{lhs} -> {' '.join(rhs)} [{p:.3f}]")

