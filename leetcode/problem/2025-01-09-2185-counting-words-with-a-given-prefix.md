---
layout: leetcode-entry
title: "2185. Counting Words With a Given Prefix"
permalink: "/leetcode/problem/2025-01-09-2185-counting-words-with-a-given-prefix/"
leetcode_ui: true
entry_slug: "2025-01-09-2185-counting-words-with-a-given-prefix"
---

[2185. Counting Words With a Given Prefix](https://leetcode.com/problems/counting-words-with-a-given-prefix/description/) easy
[blog post](https://leetcode.com/problems/counting-words-with-a-given-prefix/solutions/6253530/kotlin-rust-by-samoylenkodmitry-dz1s/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09012025-2185-counting-words-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vou1HeSppHc)
[deep-dive](https://notebooklm.google.com/notebook/f178b954-9a22-4606-91d6-0905671c671f/audio)
![1.webp](/assets/leetcode_daily_images/b4797e16.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/859

#### Problem TLDR

Count words with prefix #easy

#### Intuition

Brute-force is optimal.

#### Approach

* how short can it be?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun prefixCount(words: Array<String>, pref: String) =
        words.count { it.startsWith(pref) }

```
```rust

    pub fn prefix_count(words: Vec<String>, pref: String) -> i32 {
        words.iter().filter(|w| w.starts_with(&pref)).count() as _
    }

```
```c++

    int prefixCount(vector<string>& words, string pref) {
        int r = 0;
        for (auto &w: words) r += w.starts_with(pref);
        return r;
    }

```

