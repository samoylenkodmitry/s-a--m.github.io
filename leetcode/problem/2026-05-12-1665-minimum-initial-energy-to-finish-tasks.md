---
layout: leetcode-entry
title: "1665. Minimum Initial Energy to Finish Tasks"
permalink: "/leetcode/problem/2026-05-12-1665-minimum-initial-energy-to-finish-tasks/"
leetcode_ui: true
entry_slug: "2026-05-12-1665-minimum-initial-energy-to-finish-tasks"
---

[1665. Minimum Initial Energy to Finish Tasks](https://leetcode.com/problems/minimum-initial-energy-to-finish-tasks/solutions/8199683/kotlin-rust-by-samoylenkodmitry-237h/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12052026-1665-minimum-initial-energy?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ITPITYhawuc)

https://dmitrysamoylenko.com/leetcode/

![12.05.2026.webp](/assets/leetcode_daily_images/12.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1357

#### Problem TLDR

Energy for all tasks (spend, limit)

#### Intuition

```j
    // we can do binary search
    // but how to optimally use the energy?
    // do we know that order is always decrease t[][1]?
    //
    // i don't have any better idea, let's write bs

    // 32
    // 10-12, 10-11, 8-9, 2-4, 1-3
    //        22     12   4    2     so this is the corner case
    //
    // 1-3 2-4 10-11 10-12 8-9
    // 32  31  29    19    9       this is the optimal order
    //
    // 26 minute, hints: Figure a sorting pattern
    //                   is exactly what i can't do
    //
    // so this is a brainteaser about sorting
    //
```
Didn't solve without a hint.
Sort by (spend-limit). Then do a binary search with forward pass and energy consumption or just a backward pass and energy max(spend, limit).
The intuition behind (spend-limit): it is a greedy assumption that minimizing the *consumption* works.
Why the pair of (spend,-limit) or (-limit, spend) doesn't work? Because we want to maximize the "refund" (what was required - what was returned back).

#### Approach

* rust itertools allows one-liner

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun minimumEffort(t: Array<IntArray>) = t
        .sortedBy { it[1]-it[0] }
        .fold(0) { e, (a,b) -> max(e+a, b) }
```
```rust
    pub fn minimum_effort(t: Vec<Vec<i32>>) -> i32 {
        t.iter().sorted_by_key(|v|v[1]-v[0])
        .fold(0, |e, v| v[1].max(e+v[0]))
    }
```

