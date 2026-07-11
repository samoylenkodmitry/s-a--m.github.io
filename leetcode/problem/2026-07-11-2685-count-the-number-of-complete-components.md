---
layout: leetcode-entry
title: "2685. Count the Number of Complete Components"
permalink: "/leetcode/problem/2026-07-11-2685-count-the-number-of-complete-components/"
leetcode_ui: true
entry_slug: "2026-07-11-2685-count-the-number-of-complete-components"
---

[2685. Count the Number of Complete Components](https://leetcode.com/problems/count-the-number-of-complete-components/solutions/8390139/kotlin-rust-by-samoylenkodmitry-lwl3/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11072026-2685-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pmytBsVvW-A)

https://dmitrysamoylenko.com/leetcode/

![11.07.2026.webp](/assets/leetcode_daily_images/11.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1417

#### Problem TLDR

Fully connected components

#### Intuition

1. Union-Find to find connected components
2. Count incoming edges for each node
3. Fully connected group size is the number of edges for each node +1

#### Approach

* 50 elements can fit into long variable
* adjacency matrix would look the same for each group

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun countCompleteComponents(n: Int, e: Array<IntArray>) = e
        .fold(LongArray(n){1L shl it}){ m, (a, b) -> m[a] += 1L shl b; m[b] += 1L shl a; m }
        .groupBy { it }.count { (k, v) -> v.size == k.countOneBits() }
```
```rust
6
```

