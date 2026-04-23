---
layout: leetcode-entry
title: "3653. XOR After Range Multiplication Queries I"
permalink: "/leetcode/problem/2026-04-08-3653-xor-after-range-multiplication-queries-i/"
leetcode_ui: true
entry_slug: "2026-04-08-3653-xor-after-range-multiplication-queries-i"
---

[3653. XOR After Range Multiplication Queries I]() medium
[blog post](https://leetcode.com/problems/xor-after-range-multiplication-queries-i/solutions/7823046/kotlin-rust-by-samoylenkodmitry-i69e/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08042026-3653-xor-after-range-multiplication?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EaoFDMYzxpQ)

![08.04.2026.webp](/assets/leetcode_daily_images/08.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1322

#### Problem TLDR

Run queries #medium

#### Intuition

Just run queries.

#### Approach

* cast to long in-place

#### Complexity

- Time complexity:
$$O(nq)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 96ms
    fun xorAfterQueries(n: IntArray, q: Array<IntArray>) = n.apply {
        for ((l,r,k,v) in q) for (i in l..r step k)
            n[i] = ((1L*n[i] * v) % 1000000007).toInt()
    }.reduce(Int::xor)
```
```rust
// 56ms
    pub fn xor_after_queries(mut n: Vec<i32>, q: Vec<Vec<i32>>) -> i32 {
        for v in q { for i in (v[0] as usize..=v[1] as _).step_by(v[2] as _) {
            n[i] = (n[i] as i64 * v[3] as i64 % 1000000007) as _
        }}
        n.into_iter().fold(0, |a, b| a ^ b)
    }
```

