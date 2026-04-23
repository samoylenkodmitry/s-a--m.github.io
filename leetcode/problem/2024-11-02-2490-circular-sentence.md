---
layout: leetcode-entry
title: "2490. Circular Sentence"
permalink: "/leetcode/problem/2024-11-02-2490-circular-sentence/"
leetcode_ui: true
entry_slug: "2024-11-02-2490-circular-sentence"
---

[2490. Circular Sentence](https://leetcode.com/problems/circular-sentence/description/) easy
[blog post](https://leetcode.com/problems/circular-sentence/solutions/5996804/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02112024-2490-circular-sentence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_HnG4GtXQiM)
[deep-dive](https://notebooklm.google.com/notebook/9969bc2e-289d-4344-82a6-89e0b6c9195e/audio)
![1.webp](/assets/leetcode_daily_images/73b3ec13.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/788

#### Problem TLDR

Are words circular? #easy

#### Intuition

If the current char is space, check its surroundings. Don't forget to check the first and the last letter of the entire sentence (that was what I forgot)

#### Approach

* let's do codegolf
* windows() is nice
* regex is slow (and a separate kind of language, but powerful if mastered)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun isCircularSentence(sentence: String) =
        sentence[0] == sentence.last() &&
        sentence.windowed(3).all { it[1] != ' ' || it[0] == it[2] }

```
```rust

    pub fn is_circular_sentence(sentence: String) -> bool {
        let bs = sentence.as_bytes();
        bs[0] == bs[bs.len() - 1] && bs.windows(3)
            .all(|w| w[1] != b' ' || w[0] == w[2])
    }

```
```c++

    bool isCircularSentence(string sentence) {
        return sentence[0] == sentence.back() &&
            !regex_search(sentence, regex("(.) (?!\\1)"));
    }

```

