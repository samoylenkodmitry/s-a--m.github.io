---
layout: leetcode-entry
title: "3110. Score of a String"
permalink: "/leetcode/problem/2024-06-01-3110-score-of-a-string/"
leetcode_ui: true
entry_slug: "2024-06-01-3110-score-of-a-string"
---

[3110. Score of a String](https://leetcode.com/problems/score-of-a-string/description/) easy
[blog post](https://leetcode.com/problems/score-of-a-string/solutions/5238663/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01062024-3110-score-of-a-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/lLVA9OcHKvw)
![2024-06-01_08-37.webp](/assets/leetcode_daily_images/eed71ea0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/625

#### Problem TLDR

Sum(abs(window)) #easy

#### Intuition

Just do what is asked. Use iterators preferably.

#### Approach

Some notes to Rust:
* `as_bytes` gives a slice of [u8] and slices have a `window`
* there is an `abs_diff`, can save some symbols

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun scoreOfString(s: String): Int =
        s.windowed(2).sumBy { abs(it[0] - it[1]) }

```
```rust

    pub fn score_of_string(s: String) -> i32 {
        s.as_bytes().windows(2)
        .map(|x| x[0].abs_diff(x[1]) as i32).sum()
    }

```

