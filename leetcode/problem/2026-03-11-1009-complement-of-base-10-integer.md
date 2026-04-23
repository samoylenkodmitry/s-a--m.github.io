---
layout: leetcode-entry
title: "1009. Complement of Base 10 Integer"
permalink: "/leetcode/problem/2026-03-11-1009-complement-of-base-10-integer/"
leetcode_ui: true
entry_slug: "2026-03-11-1009-complement-of-base-10-integer"
---

[1009. Complement of Base 10 Integer](https://open.substack.com/pub/dmitriisamoilenko/p/11032026-1009-complement-of-base?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/11032026-1009-complement-of-base?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11032026-1009-complement-of-base?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/lAA3Q_j5X2M)

![f0a013aa-d52d-4f76-9773-52e129098a53 (1).webp](/assets/leetcode_daily_images/e00fe607.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1294

#### Problem TLDR

Invert binary #easy

#### Intuition

Use a bitmask of the next power of two - 1.

#### Approach

* Kotlin: takeHighestOneBit, countLeadingZeroBits
* Rust: leasing_zeros, next_power_of_two, ilog2
* 0 case trick: use |1 to check bits count at least 1

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 0ms
    fun bitwiseComplement(n: Int) =
    (n or 1).takeHighestOneBit()*2 - n - 1
```
```rust
// 0ms
    pub fn bitwise_complement(n: i32) -> i32 {
        n^(2<<(n|1).ilog2())-1
    }
```

