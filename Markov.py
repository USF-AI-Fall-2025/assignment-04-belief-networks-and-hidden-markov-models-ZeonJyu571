import math
import string

###################
# 1. Preparations #
###################

def parsing(file_path):
    """
    outputs:
        (1) a list of tuples containing all the pairs of correct and typed words
        (2) a set of all correct words
    """
    pairs = []
    correct_words_list = set()
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            correct, wrongs = line.split(":", 1)
            correct = correct.strip().lower()
            correct_words_list.add(correct)
            for wrong in wrongs.strip().split():
                pairs.append((correct, wrong.lower()))
    return pairs, sorted(correct_words_list)


def align(s1, s2):
    """
    Using Levenstein algorithm to find the optimal alignment of the correct word and the typed word.
    "-" marks a gap in the returned result.
    """
    n, m = len(s1), len(s2)
    dp = [[0]*(m + 1) for _ in range(n + 1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    # fill the table and ascertain the minimum cost for alignment
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # delete
                dp[i][j - 1] + 1,       # insert
                dp[i - 1][j - 1] + cost   # replace / match
            )

    # backtrace
    i, j = n, m
    a1 = []
    a2 = []
    while i>0 or j>0:
        if i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + (0 if s1[i-1]==s2[j-1] else 1):
            a1.append(s1[i-1])
            a2.append(s2[j-1])
            i -= 1
            j -= 1
        elif i>0 and dp[i][j] == dp[i-1][j] + 1:
            a1.append(s1[i-1])
            a2.append("-")
            i -= 1
        else:
            a1.append("-")
            a2.append(s2[j-1])
            j -= 1

    return "".join(reversed(a1)), "".join(reversed(a2))


###########################
# 2. Emission Probability #
###########################

def find_emission(aligned_pairs, hidden_states):
    """
    For missing letter in the typed word, the emission will be correct letter -> "-".

    For extra letter in the typed word, the extra letter is viewed as emitted by the previous correct letter. For example, for cat: caat, if the outcome after
    alignment is ca-t: caat, the mapping for "-" in the correct word would be a -> aa. If the "-" locates at the start, the extra letter will be viewed as 
    emitted by the special state <S>, which means the start. For example, for apple: aapple (-apple: aapple), the mapping would be <S> -> a. When there are
    consecutive "-" in the aligned correct word, they will still be handled on an one-to-one basis, and attributed to the last correct letter.

    To avoid zero probabilities, I smoothed the emission probabilities using Laplace smoothing.
    """
    observed = hidden_states + ["-"]

    counts = {}
    for c in hidden_states + ["<S>", "-"]:
        counts[c] = {}
        for x in observed:
            counts[c][x] = 0

    for c_aligned, t_aligned in aligned_pairs:
        last_real = "<S>"
        for c, t in zip(c_aligned, t_aligned):
            # Single out the case of insertion
            if c != "-":
                last_real = c
                counts[c][t] += 1
            else:
                counts[last_real][t] += 1

    # Laplace smoothing
    emission = {}
    for c in counts:
        emission[c] = {}
        total = sum(counts[c].values()) + len(observed)
        for x in observed:
            emission[c][x] = (counts[c][x] + 1) / total

    return emission

#############################
# 3. Transition Probability #
#############################

def find_transition(correct_words, char_states):

    all_states = ["<S>"] + char_states + ["<E>"]

    counts = {}
    for c1 in all_states:
        counts[c1] = {}
        for c2 in all_states:
            counts[c1][c2] = 0

    for word in correct_words:
        word = word.lower()
        prev = "<S>"
        for ch in word:
            counts[prev][ch] += 1
            prev = ch
        counts[prev]["<E>"] += 1

    # smoothing
    transition = {}
    for c1 in all_states:
        transition[c1] = {}
        total = sum(counts[c1].values()) + len(all_states)
        for c2 in all_states:
            transition[c1][c2] = (counts[c1][c2] + 1) / total

    return transition



#############################
# 4. Extra Helper Functions #
#############################
"""
Given that the transition probability from one letter to another alone can not handle the case where the emission probability of a wrong letter
overpowers the transition probability, I introduced the prior probability in word level to guarantee the accuracy of the output.
"""

def find_word_prior(pairs, correct_words):
    frequency = {word: 0 for word in correct_words}

    for correct, wrong in pairs:
        frequency[correct] += 1

    total = sum(frequency.values())

    prior = {}
    for word in correct_words:
        prior[word] = (frequency[word]) / total

    return prior

def edit_distance(s1, s2):
    """
    Using Levenshtein algorithm again for candidates, before feeding it to Viterbi.
    """
    n, m = len(s1), len(s2)
    dp = [[0]*(m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0]=i
    for j in range(m + 1): dp[0][j]=j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1]==s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]

def generate_candidates(typed, correct_words, max_dist = 2):
    """
    return only those candidates within max_dist edit distance
    """
    candidates = []

    for word in correct_words:
        if abs(len(word) - len(typed)) > max_dist:
            continue
        if edit_distance(word, typed) <= max_dist:
            candidates.append(word)

    return candidates if candidates else correct_words[:] 



##############
# 7. Viterbi #
##############

def viterbi(candidate, typed, transition, emission):
    """
    Start probabilities are already included in the transition probabilities.
    """
    correct_alligned, typed_alligned = align(candidate, typed)

    p = 1.0

    prev = "<S>"
    for c in correct_alligned:
        if c != "-":
            p *= transition[prev].get(c, 1e-12) # in case of zero prob, I opt for a very small prob instead
            prev = c
    p *= transition.get(prev, {}).get("<E>", 1e-12)

    # emission: 
    last_real = "<S>"
    for c, t in zip(correct_alligned, typed_alligned):
        if c != "-":
            last_real = c
            p *= emission.get(last_real, {}).get(t, 1e-12)
        else: # insertion
            p *= emission.get(last_real, {}).get(t, 1e-12)

    return p

def correct_word(typed, correct_words, prior, transition, emission):

    candidates = generate_candidates(typed, correct_words)

    best_w = None
    best_score = -1

    for word in candidates:
        word_prob = viterbi(word, typed, transition, emission)
        score = prior[word] * word_prob
        if score > best_score:
            best_score = score
            best_w = word

    return best_w, best_score


####################
# 8. Main function #
####################

if __name__ == "__main__":

    pairs, correct_words = parsing("aspell.txt")
    hidden_states = sorted({ch for w in correct_words for ch in w})
    aligned_pairs = [align(c, t) for c, t in pairs]
    emission = find_emission(aligned_pairs, hidden_states)
    transition = find_transition(correct_words, hidden_states)
    prior = find_word_prior(pairs, correct_words)

    while True:
        line = input("\nEnter your text without punctuation: ").strip().lower()
        tokens = line.split()
        corrected_tokens = []

        for token in tokens:
            w, score = correct_word(token, correct_words, prior, transition, emission)
            corrected_tokens.append(w)
        
        corrected_line = " ".join(corrected_tokens)
        print("Corrected text:", corrected_line)
        


