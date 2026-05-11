---
layout: leetcode-entry
title: "2553. Separate the Digits in an Array"
permalink: "/leetcode/problem/2026-05-11-2553-separate-the-digits-in-an-array/"
leetcode_ui: true
entry_slug: "2026-05-11-2553-separate-the-digits-in-an-array"
---

[2553. Separate the Digits in an Array](https://leetcode.com/problems/separate-the-digits-in-an-array/solutions/8188273/kotlin-rust-by-samoylenkodmitry-rmab/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11052026-2553-separate-the-digits?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EVoFYklkbiM)

https://dmitrysamoylenko.com/leetcode/

![11.05.2026.webp](/assets/leetcode_daily_images/11.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1356

#### Problem TLDR

LIst of numbers to list of digits

#### Intuition

Convert to strings or do %10 with LInkedList/ArrayDeque or do the reverse.

#### Approach

* Kotlin: joinToString converts numbers to strings
* Rust: use flat_map or to_string/bytes

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun separateDigits(n: IntArray) =
        n.joinToString("").map{it-'0'}
```
```rust
    pub fn separate_digits(n: Vec<i32>) -> Vec<i32> {
        n.iter().map(|x|x.to_string()).collect::<String>()
        .bytes().map(|b|(b-b'0') as i32).collect()
    }
```

