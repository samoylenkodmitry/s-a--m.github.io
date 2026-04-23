---
layout: leetcode-entry
title: "1002. Find Common Characters"
permalink: "/leetcode/problem/2024-06-05-1002-find-common-characters/"
leetcode_ui: true
entry_slug: "2024-06-05-1002-find-common-characters"
---

[1002. Find Common Characters](https://leetcode.com/problems/find-common-characters/description/) easy
[blog post](https://leetcode.com/problems/find-common-characters/solutions/5261457/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05062024-1002-find-common-characters?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DHo74a78GCU)
![2024-06-05_07-42.webp](/assets/leetcode_daily_images/7972e9c9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/629

#### Problem TLDR

Common letters in words #easy

#### Intuition

We can count frequencies, then choose minimums for each char. Or do the reverse: for each char count minimum count in all words.

#### Approach

The frequencies code is faster, but the opposite approach is less verbose.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, but can be O(n) to hold the result

#### Code

```kotlin

    fun commonChars(words: Array<String>) =
        ('a'..'z').map { c ->
            List(words.minOf { it.count { it == c } }) { "$c" }
        }.flatten()

```
```rust

    pub fn common_chars(words: Vec<String>) -> Vec<String> {
        ('a'..='z').map(|c| {
            let min_cnt = words.iter().map(|w|
                w.chars().filter(|a| *a == c).count()).min();
            vec![format!("{c}"); min_cnt.unwrap_or(0)]
        }).flatten().collect()
    }

```

