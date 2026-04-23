---
layout: leetcode-entry
title: "3163. String Compression III"
permalink: "/leetcode/problem/2024-11-04-3163-string-compression-iii/"
leetcode_ui: true
entry_slug: "2024-11-04-3163-string-compression-iii"
---

[3163. String Compression III](https://leetcode.com/problems/string-compression-iii/description/) medium
[blog post](https://leetcode.com/problems/string-compression-iii/solutions/6005567/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04112024-3163-string-compression?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/P2H4ZyAixYM)
[deep-dive](https://notebooklm.google.com/notebook/ff73d536-c831-44ed-a245-e255307f5329/audio)
![1.webp](/assets/leetcode_daily_images/d4204c8a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/790

#### Problem TLDR

Compress repeating chars in string #medium #two_pointers

#### Intuition

This is all about how you implement it.
One way is to use a `counter` and analyze the current position.
Another way is to use the `two pointers` and skip all the repeating characters making a single point of appending.

#### Approach

* Rust has a cool `chunk_by` method

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun compressedString(word: String) = buildString {
        var j = 0; var i = 0
        while (i < word.length) {
            while (j < min(i + 9, word.length)
                   && word[i] == word[j]) j++
            append("${j - i}${word[i]}")
            i = j
        }
    }

```
```rust

    pub fn compressed_string(word: String) -> String {
        word.into_bytes().chunk_by(|a, b| a == b)
        .flat_map(|ch| ch.chunks(9).flat_map(|c|
            [(b'0' + c.len() as u8) as char, c[0].into()])
        ).collect()
    }

```
```c++

    string compressedString(string w) {
        string r;
        for(int i = 0, j = 0; i < size(w); i = j) {
            for(; j < i + 9 && j < size(w) && w[i] == w[j]; ++j);
            r += 48 + j - i; r += w[i];
        }
        return r;
    }

```

