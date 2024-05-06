synonyms = sorted(
    [
        syn
        for syn in word_dict[cleaned_word]
        if syn[1] < 1.0 and cleaned_word not in syn[0]
    ],
    key=lambda x: x[1],
    reverse=True,
)
