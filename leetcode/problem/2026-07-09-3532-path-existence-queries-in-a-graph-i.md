---
layout: leetcode-entry
title: "3532. Path Existence Queries in a Graph I"
permalink: "/leetcode/problem/2026-07-09-3532-path-existence-queries-in-a-graph-i/"
leetcode_ui: true
entry_slug: "2026-07-09-3532-path-existence-queries-in-a-graph-i"
---

[3532. Path Existence Queries in a Graph I](https://leetcode.com/problems/path-existence-queries-in-a-graph-i/solutions/8385720/kotlin-rust-by-samoylenkodmitry-lrvc/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09072026-3532-path-existence-queries?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/SDMKProm1Ew)

https://dmitrysamoylenko.com/leetcode/

![09.07.2026.webp](/assets/leetcode_daily_images/09.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1415

#### Problem TLDR

Queries of connected nodes

#### Intuition

* build a Union-Find, iterate once, connect consequent numbers

#### Approach

* we can reuse the nums array

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun pathExistenceQueries(n: Int, ns: IntArray, md: Int, qs: Array<IntArray>) = run {
        ns.reduceIndexed { i, p, c -> c.also { if (c - p <= md) ns[i] = ns[i-1] } }
        qs.map { (a,b) -> ns[a]==ns[b] }
    }
```
```rust
    pub fn path_existence_queries(n: i32, mut ns: Vec<i32>, md: i32, qs: Vec<Vec<i32>>) -> Vec<bool> {
        let mut p = ns[0]; for i in 1..ns.len() { let n = ns[i]; if n - p <= md { ns[i] = ns[i-1]}; p = n }
        qs.iter().map(|q|ns[q[0] as usize]==ns[q[1] as usize]).collect()
    }
```

