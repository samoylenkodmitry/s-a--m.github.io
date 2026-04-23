---
layout: leetcode-entry
title: "1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence"
permalink: "/leetcode/problem/2024-12-02-1455-check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence/"
leetcode_ui: true
entry_slug: "2024-12-02-1455-check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence"
---

[1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence](https://leetcode.com/problems/check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence/description/) easy
[blog post](https://leetcode.com/problems/check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence/solutions/6103399/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02122024-1455-check-if-a-word-occurs?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/P_77P7z4ew4)
[deep-dive](https://notebooklm.google.com/notebook/d201f718-1e5c-4d7c-946d-f60632a42142/audio)
![1.webp](/assets/leetcode_daily_images/5b7fae70.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/819

#### Problem TLDR

Position of the prefix #easy

#### Intuition

The O(n) time and O(1) memory solution is possible (see c++).

#### Approach

* we can prepend a word to shorten the index adjusting logic
* c++ will shoot in your foot for comparing `-1` with `.size()`
* rust has a nice `.map_or`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

    fun isPrefixOfWord(sentence: String, searchWord: String) =
        .indexOfFirst { it.startsWith(searchWord) }

```
```rust

    pub fn is_prefix_of_word(sentence: String, search_word: String) -> i32 {
        sentence.split_whitespace().position(|w| w.starts_with(&search_word))
        .map_or(-1, |i| 1 + i as i32)
    }

```
```c++

    int isPrefixOfWord(string s, string w) {
        int p = 1, j = 0, n = w.size();
        for (int i = 0; i < s.size() && j < n; ++i)
            s[i] == ' ' ? j = 0, ++p :
            j >= 0 && s[i] == w[j] ? ++j : j = -1;
        return j < n ? -1 : p;
    }

```

