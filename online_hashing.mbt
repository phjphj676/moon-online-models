///|
/// Token hashing utilities for text and event-key features.
pub struct HashingVectorizer {
  hasher : FeatureHasher
  lowercase : Bool
  ngram_order : Int
}

///|
pub fn HashingVectorizer::new(
  buckets : Int,
  lowercase? : Bool = true,
  ngram_order? : Int = 1,
) -> HashingVectorizer {
  {
    hasher: FeatureHasher::new(buckets),
    lowercase,
    ngram_order: if ngram_order < 1 {
      1
    } else {
      ngram_order
    },
  }
}

///|
pub fn HashingVectorizer::tokens(
  self : HashingVectorizer,
  text : String,
) -> Array[String] {
  let normalized = if self.lowercase { text.to_lower() } else { text }
  normalized
  .split(" ")
  .filter(value => !value.is_empty())
  .map(value => value.to_owned())
  .to_array()
}

///|
pub fn HashingVectorizer::encode(
  self : HashingVectorizer,
  text : String,
) -> SparseVector {
  let words = self.tokens(text)
  if self.ngram_order == 1 {
    self.hasher.encode(words)
  } else {
    self.hasher.encode(generate_ngrams(words, self.ngram_order))
  }
}

///|
fn generate_ngrams(words : Array[String], order : Int) -> Array[String] {
  let result = Array::make(0, "")
  for start in 0..<words.length() {
    let mut value = ""
    for offset in 0..<order {
      let index = start + offset
      if index >= words.length() {
        break
      }
      value = if value == "" {
        words[index]
      } else {
        "\{value}_\{words[index]}"
      }
      result.push(value)
    }
  }
  result
}

///|
pub fn HashingVectorizer::buckets(self : HashingVectorizer) -> Int {
  self.hasher.buckets()
}

///|
pub struct TextStatistics {
  mut documents : Int
  mut tokens : Int
  vocabulary : Map[String, Int]
}

///|
pub fn TextStatistics::new() -> TextStatistics {
  { documents: 0, tokens: 0, vocabulary: {} }
}

///|
pub fn TextStatistics::observe(
  self : TextStatistics,
  words : Array[String],
) -> Unit {
  self.documents += 1
  self.tokens += words.length()
  for word in words {
    self.vocabulary.update_or_default(word, 0, previous => previous + 1)
  }
}

///|
pub fn TextStatistics::documents(self : TextStatistics) -> Int {
  self.documents
}

///|
pub fn TextStatistics::tokens(self : TextStatistics) -> Int {
  self.tokens
}

///|
pub fn TextStatistics::vocabulary_size(self : TextStatistics) -> Int {
  self.vocabulary.length()
}

///|
pub fn TextStatistics::frequency(self : TextStatistics, word : String) -> Int {
  self.vocabulary.get(word).unwrap_or(0)
}

///|
pub fn TextStatistics::top_words(
  self : TextStatistics,
  k : Int,
) -> Array[String] {
  let entries = self.vocabulary.to_array()
  entries.sort_by((left, right) => {
    if left.1 > right.1 {
      -1
    } else if left.1 < right.1 {
      1
    } else {
      String::compare(left.0, right.0)
    }
  })
  let limit = if k < 0 {
    0
  } else if k > entries.length() {
    entries.length()
  } else {
    k
  }
  Array::makei(limit, i => entries[i].0)
}

///|
pub fn TextStatistics::reset(self : TextStatistics) -> Unit {
  self.documents = 0
  self.tokens = 0
  self.vocabulary.clear()
}
