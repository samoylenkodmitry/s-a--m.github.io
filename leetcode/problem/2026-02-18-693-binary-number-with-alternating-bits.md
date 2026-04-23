---
layout: leetcode-entry
title: "693. Binary Number with Alternating Bits"
permalink: "/leetcode/problem/2026-02-18-693-binary-number-with-alternating-bits/"
leetcode_ui: true
entry_slug: "2026-02-18-693-binary-number-with-alternating-bits"
---

[693. Binary Number with Alternating Bits](https://open.substack.com/pub/dmitriisamoilenko/p/18022026-693-binary-number-with-alternating?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/18022026-693-binary-number-with-alternating?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18022026-693-binary-number-with-alternating?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pot5OUHEMfc)

![c2a476b9-d0a9-4fc9-b2d3-646f51977522 (1).webp](/assets/leetcode_daily_images/65fa3b6b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1273

#### Problem TLDR

Alternating bits #easy #bits

#### Intuition

Check 31 bits with brute force.

The "clever solutions":
* shift right by 2 positions, should match; shift right by 1 positions, should be opposite
* shift right by 1 position, do or: should be all ones

#### Approach

* regex: (00|11)

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 0ms
    fun hasAlternatingBits(n: Int) =
    n or n/4 == n && n and n/2 == 0
    /*
    Regex("(11|00)") !in n.toString(2)
    (n xor n/2).toString(2).all { it == '1' }
     */
```
```rust
// 0ms
    pub fn has_alternating_bits(mut n: i32) -> bool {
        n ^= n/2; n & n+1 == 0
    }
```

