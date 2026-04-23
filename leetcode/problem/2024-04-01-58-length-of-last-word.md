---
layout: leetcode-entry
title: "58. Length of Last Word"
permalink: "/leetcode/problem/2024-04-01-58-length-of-last-word/"
leetcode_ui: true
entry_slug: "2024-04-01-58-length-of-last-word"
---

[58. Length of Last Word](https://leetcode.com/problems/length-of-last-word/description/) easy
[blog post](https://leetcode.com/problems/length-of-last-word/solutions/4955206/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01042024-58-length-of-last-word?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/YaRWWIW6Krw)
![2024-04-01_08-06.webp](/assets/leetcode_daily_images/f2fd0935.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/557

#### Problem TLDR

Last word length #easy

#### Intuition

There are many ways, let's try to write an efficient solution.
Iterate from the end, stop after the first word.

#### Approach

In Kotlin we can use `first`, `takeWhile` and `count`.
In Rust let's to write a simple `for` loop over `bytes`.

#### Complexity

- Time complexity:
$$O(w + b)$$, where `w` is a last word length, and `b` suffix blank space length

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun lengthOfLastWord(s: String) =
    ((s.lastIndex downTo 0).first { s[it] > ' ' } downTo 0)
    .asSequence().takeWhile { s[it] > ' ' }.count()

```
```rust

  pub fn length_of_last_word(s: String) -> i32 {
    let mut c = 0;
    for b in s.bytes().rev() {
      if b > b' ' { c += 1 } else if c > 0 { return c }
    }
    c
  }

```

