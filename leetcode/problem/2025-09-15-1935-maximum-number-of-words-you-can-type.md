---
layout: leetcode-entry
title: "1935. Maximum Number of Words You Can Type"
permalink: "/leetcode/problem/2025-09-15-1935-maximum-number-of-words-you-can-type/"
leetcode_ui: true
entry_slug: "2025-09-15-1935-maximum-number-of-words-you-can-type"
---

[1935. Maximum Number of Words You Can Type](https://leetcode.com/problems/maximum-number-of-words-you-can-type/description) easy
[blog post](https://leetcode.com/problems/maximum-number-of-words-you-can-type/solutions/7192116/kotlin-rust-by-samoylenkodmitry-bxzu/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15092025-1935-maximum-number-of-words?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/K-fzBO3LVxY)

![1.webp](/assets/leetcode_daily_images/f6c9189b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1113

#### Problem TLDR

Count words with all letters #easy

#### Intuition

No special algo here. Broken letters is up to 26, no hashset needed.

#### Approach

* write a one-liner

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 14ms
    fun canBeTypedWords(txt: String, bl: String) =
        txt.split(" ").count { it.all { it !in bl}}

```
```rust

// 0ms
    pub fn can_be_typed_words(txt: String, bl: String) -> i32 {
        txt.split(" ").filter(|w| !w.chars().any(|c| bl.contains(c))).count() as _
    }

```

