---
layout: leetcode-entry
title: "648. Replace Words"
permalink: "/leetcode/problem/2024-06-07-648-replace-words/"
leetcode_ui: true
entry_slug: "2024-06-07-648-replace-words"
---

[648. Replace Words](https://leetcode.com/problems/replace-words/description/) medium
[blog post](https://leetcode.com/problems/replace-words/solutions/5272240/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07062024-648-replace-words?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FAnZD16Ltw4)
![2024-06-07_07-09_1.webp](/assets/leetcode_daily_images/85ed0fcd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/632

#### Problem TLDR

Replace words with suffixes from dictionary #medium #trie

#### Intuition

Walk through the word and check if the suffix is in the dictionary. To speed up this we can use a HashMap or a Trie.

#### Approach

Let's use both HashMap and Trie. HashMap code is shorter but slower.

#### Complexity

- Time complexity:
$$O(n)$$, O(nw^2) for HashMap solution, as we rebuilding each suffix in the word of `w` length

- Space complexity:
$$O(d + w)$$

#### Code

```kotlin

    fun replaceWords(dictionary: List<String>, sentence: String): String {
        class Trie(var word: Int = -1): HashMap<Char, Trie>()
        val trie = Trie()
        for ((i, r) in dictionary.withIndex()) {
            var t = trie
            for (c in r) t = t.getOrPut(c) { Trie() }
            t.word = i
        }
        return sentence.split(" ").map {
            var t = trie
            for (c in it) {
                if (t.word >= 0) break
                t = t[c] ?: break
            }
            dictionary.getOrNull(t.word) ?: it
        }.joinToString(" ")
    }

```
```rust

    pub fn replace_words(dictionary: Vec<String>, sentence: String) -> String {
        let set = dictionary.into_iter().collect::<HashSet<_>>();
        sentence.split(" ").map(|s| {
            let mut w = String::new();
            for c in s.chars() {
                w.push(c);
                if set.contains(&w) { break }
            }; w
        }).collect::<Vec<_>>().join(" ")
    }

```

