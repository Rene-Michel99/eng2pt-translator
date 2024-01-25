import tensorflow as tf


def preprocess_pt(text):
    # Handle specific accents that have important grammatical mean
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, " é ", ' ee ')
    text = tf.strings.regex_replace(text, " à ", ' aa ')

    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def preprocess_eng(text):
    text = tf.strings.lower(text)

    text = tf.strings.regex_replace(text, "i'm", 'i am')
    text = tf.strings.regex_replace(text, "it's", 'it is')
    text = tf.strings.regex_replace(text, "where's", 'where is')
    text = tf.strings.regex_replace(text, "what's", 'what is')
    text = tf.strings.regex_replace(text, "why's", 'why is')
    text = tf.strings.regex_replace(text, "he's", 'he is')
    text = tf.strings.regex_replace(text, "how's", 'how is')
    text = tf.strings.regex_replace(text, "here's", 'here is')
    text = tf.strings.regex_replace(text, "let's", 'let us')
    text = tf.strings.regex_replace(text, "'re", ' are')
    text = tf.strings.regex_replace(text, "'d", ' had')
    text = tf.strings.regex_replace(text, "n't", ' not')
    text = tf.strings.regex_replace(text, "'ve", ' have')
    text = tf.strings.regex_replace(text, "'ll", ' will')
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    return text
