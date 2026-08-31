---
layout: leetcode-entry
title: "2058. Find the Minimum and Maximum Number of Nodes Between Critical Points"
permalink: "/leetcode/problem/2026-08-31-2058-find-the-minimum-and-maximum-number-of-nodes-between-critical-points/"
leetcode_ui: true
entry_slug: "2026-08-31-2058-find-the-minimum-and-maximum-number-of-nodes-between-critical-points"
---

[2058. Find the Minimum and Maximum Number of Nodes Between Critical Points](https://leetcode.com/problems/find-the-minimum-and-maximum-number-of-nodes-between-critical-points/solutions/8492781/kotlin-rust-by-samoylenkodmitry-yoz9/) medium
[substack](https://dmitriisamoilenko.substack.com/p/31082026-2058-find-the-minimum-and?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/qeMjpOveXxY)

https://dmitrysamoylenko.com/leetcode/

![31.08.2026.webp](/assets/leetcode_daily_images/31.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1468

#### Problem TLDR

Min and max distance between extremums

#### Intuition

Iterate, find extremums, track the position.

#### Approach

* use the sequence
* Rust itertools unfold

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun nodesBetweenCriticalPoints(h: ListNode?) =
    generateSequence(h) { it.next }.map { it.`val` }.windowed(3)
    .mapIndexedNotNull { i, (a, b, c) -> i.takeIf { 1L*(a-b)*(c-b) > 0 }}
    .toList().run {
        if (size < 2) listOf(-1, -1)
        else listOf(-zipWithNext(Int::minus).max(), last()-first())
    }
```
```rust
    pub fn nodes_between_critical_points(h: Option<Box<ListNode>>) -> Vec<i32> {
        let v: Vec<_> = unfold(h, |n| n.take().map(|b| { *n = b.next; b.val }))
            .tuple_windows().positions(|(a, b, c)| (a-b)as i64*(c-b)as i64>0)
            .map(|i| i as i32).collect();
        if v.len() < 2 { vec![-1, -1] } else
        { vec![v.windows(2).map(|w| w[1] - w[0]).min().unwrap(), v.last().unwrap() - v[0]] }
    }
```

