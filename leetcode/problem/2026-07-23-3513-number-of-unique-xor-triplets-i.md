---
layout: leetcode-entry
title: "3513. Number of Unique XOR Triplets I"
permalink: "/leetcode/problem/2026-07-23-3513-number-of-unique-xor-triplets-i/"
leetcode_ui: true
entry_slug: "2026-07-23-3513-number-of-unique-xor-triplets-i"
---

[3513. Number of Unique XOR Triplets I](https://leetcode.com/problems/number-of-unique-xor-triplets-i/solutions/8414589/kotlin-rust-by-samoylenkodmitry-npoq/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23072026-3513-number-of-unique-xor?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/PNn1JVcV7wM)

https://dmitrysamoylenko.com/leetcode/

![23.07.2026.webp](/assets/leetcode_daily_images/23.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1429

#### Problem TLDR

Uniq xor of tripplets of 1..n

#### Intuition

```j
    // for xor order doesnt matter
    // nums are the same 1..n
    // 1 xor 1 = 0
    // 1 ^ 0 = 1
    // 1 ^ 0 ^ 1 = 0
    // 1 ^ 0 ^ 0 = 1
    // 1 ^ 1 ^ 1 = 1
    // 1 ^ 1 ^ 0 = 0
    // uniq?
```
Order doesnt matter. The maximum value 0b1xxxxx allows to construct any intermediate value by taking 1 xor 1 xor n[i] plus tail bits can construct any value of up to 2*max.

#### Approach

* simulate and see the results, then derive whats happening

#### Complexity

- Time complexity:
$$O(logn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun uniqueXorTriplets(n: IntArray) =
    if (n.size <3) n.size else 2*n.size.takeHighestOneBit()
```
```rust
    pub fn unique_xor_triplets(n: Vec<i32>) -> i32 {
        let l = n.len() as i32; if l < 3 { l } else { 1 << (32 - l.leading_zeros()) }
    }
```

