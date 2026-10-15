---
layout: leetcode-entry
title: "2913. Subarrays Distinct Element Sum of Squares I"
permalink: "/leetcode/problem/2026-10-15-2913-subarrays-distinct-element-sum-of-squares-i/"
leetcode_ui: true
entry_slug: "2026-10-15-2913-subarrays-distinct-element-sum-of-squares-i"
---

[2913. Subarrays Distinct Element Sum of Squares I](https://leetcode.com/problems/subarrays-distinct-element-sum-of-squares-i/solutions/8573830/kotlin-rust-by-samoylenkodmitry-5omy/) easy
[substack](https://dmitriisamoilenko.substack.com/p/15102026-2913-subarrays-distinct?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/nu5lsK8c3ZM)

https://dmitrysamoylenko.com/leetcode/

![15.10.2026.webp](/assets/leetcode_daily_images/15.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1512

#### Problem TLDR

Sum of squares of uniq numbers counts in all subarrays

#### Intuition

Brute force by taking each start, each end and hashset size between them.

#### Approach

* or try all sliding windows
* or reuse hashset/bitmask for O(n^2)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun sumCounts(n: List<Int>)=
    (1..n.size).sumOf{n.windowed(it){val s=it.toSet().size;s*s}.sum()}
```
```rust
    pub fn sum_counts(n: Vec<i32>) -> i32 {
        let mut a=0;for i in 0..n.len(){let mut m=0u128;for x in&n[i..]{m|=1<<x;a+=m.count_ones().pow(2)}}a as _
    }
```

