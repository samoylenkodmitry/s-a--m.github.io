---
layout: leetcode-entry
title: "1408. String Matching in an Array"
permalink: "/leetcode/problem/2025-01-07-1408-string-matching-in-an-array/"
leetcode_ui: true
entry_slug: "2025-01-07-1408-string-matching-in-an-array"
---

[1408. String Matching in an Array](https://leetcode.com/problems/string-matching-in-an-array/description/) easy
[blog post](https://leetcode.com/problems/string-matching-in-an-array/solutions/6243721/kotlin-rust-by-samoylenkodmitry-s0hg/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07012025-1408-string-matching-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UHgFY3seZCo)
[deep-dive](https://notebooklm.google.com/notebook/ad5d34b7-a8d1-4f14-a324-8e099d2d7408/audio)
![1.webp](/assets/leetcode_daily_images/bfd761e8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/857

#### Problem TLDR

All substrings #easy

#### Intuition

Brute force is accepted.

#### Approach

* we can improve speed by searching for at least 2 matches in the joined words (and speed this up with KMP or Robin-Karp rolling hash)
* careful to not include the word twice

#### Complexity

- Time complexity:
$$O(n^2w^2)$$, w^2 for `word1.contains(word2)`

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun stringMatching(words: Array<String>) =
        words.filter { w -> words.any { w != it && w in it }}

```
```rust

    pub fn string_matching(words: Vec<String>) -> Vec<String> {
        words.iter().filter(|w| words.iter().any(|w2|
            *w != w2 && w2.contains(*w))).cloned().collect()
    }

```
```c++

    vector<string> stringMatching(vector<string>& words) {
        vector<string> r;
        for (int i = 0; i < words.size(); ++i)
            for (int j = 0; j < words.size(); ++j)
                if (i != j && words[j].find(words[i]) != string::npos) {
                    r.push_back(words[i]); break;
                }
        return r;
    }

```

