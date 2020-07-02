from tqdm import tqdm


if __name__ == "__main__":
  
  with open("resources/word_embeddings/glove.txt") as inp:
    lines = inp.readlines()

  sentences = ['I like cats and other animals',
                'I want to live in New York']

  words = set()
  for s in sentences:
    words = words.union(set(s.split(' ')))

  i = 0
  bar = tqdm(total = len(lines))

  with open("resources/word_embeddings/toy_glove.txt", "w") as out:
    while(words):
      if lines[i].split(' ')[0] in words:
        out.write(lines[i])
        words.remove(lines[i].split(' ')[0])
        print('"{}" found, {} words remaining'.format(lines[i].split(' ')[0], len(words)))
      i += 1
      bar.update(1)