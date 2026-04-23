---
layout: leetcode-entry
title: "3314. Construct the Minimum Bitwise Array I"
permalink: "/leetcode/problem/2026-01-20-3314-construct-the-minimum-bitwise-array-i/"
leetcode_ui: true
entry_slug: "2026-01-20-3314-construct-the-minimum-bitwise-array-i"
---

[3314. Construct the Minimum Bitwise Array I](https://leetcode.com/problems/construct-the-minimum-bitwise-array-i/description) easy
[blog post](https://leetcode.com/problems/construct-the-minimum-bitwise-array-i/solutions/7509292/kotlin-rust-by-samoylenkodmitry-3wq2/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20012026-3314-construct-the-minimum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/nPIRhjJj8zI)

![a4f2597c-ea61-4722-ba55-8133a91c435f (1).webp](/assets/leetcode_daily_images/d23b594e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1243

#### Problem TLDR

Reverse x | (x+1) operation on prime numbers #easy

#### Intuition

The op: x|(x+1) does set rightmost 0 to 1. 100: 100 | 101 = 101
Reversing it: set first suffix 1 bit to 0. 101: 100

```j
    // 111 +1
    //1000
    //
    // 101 100 or 101
    // 111  11 or 100
    //  10  -
    //  11   1 or 10
    //
    //  1011 1001 or 1010
    //  1101 1100 or 1
    // 11111 1111 or 10000
```

#### Approach

* do a brute-force
* learn from others

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 7ms
    fun minBitwiseArray(n: List<Int>)=n.map {
        if (it == 2) -1 else it xor it.inv().takeLowestOneBit()/2
    }
```
```rust
// 0ms
    pub fn min_bitwise_array(n: Vec<i32>) -> Vec<i32> {
        n.iter().map(|&n| if n == 2 {-1} else {n^((n+1)&!n)/2}).collect()
    }
```

