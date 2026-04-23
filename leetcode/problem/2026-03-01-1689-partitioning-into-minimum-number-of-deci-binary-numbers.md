---
layout: leetcode-entry
title: "1689. Partitioning Into Minimum Number Of Deci-Binary Numbers"
permalink: "/leetcode/problem/2026-03-01-1689-partitioning-into-minimum-number-of-deci-binary-numbers/"
leetcode_ui: true
entry_slug: "2026-03-01-1689-partitioning-into-minimum-number-of-deci-binary-numbers"
---

[1689. Partitioning Into Minimum Number Of Deci-Binary Numbers](https://open.substack.com/pub/dmitriisamoilenko/p/01022026-1689-partitioning-into-minimum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/01022026-1689-partitioning-into-minimum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01022026-1689-partitioning-into-minimum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5kz6Fk-eQVA)

![e1f08fed-fa61-41b4-b9c2-128ee956ab97 (1).webp](/assets/leetcode_daily_images/42aafbe1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1284

#### Problem TLDR

Min 01-strings add up to target #medium

#### Intuition

```j
    // 279
    // 101
    // 101
    //  11 * 7
    // 2
    //  7
    //   (2+7)
    //
    // 12345
    // 11111
    //  1111
    //   111
    //    11
    //     1
    //
    // 54321
    // 11111
    // 1111
    // 111
    // 11
    // 1
    //
    // 105
    //
    // 50
    //
    // 505
```

Greedy idea: take as big binary number as possible to quickly fill to target.

#### Approach

* max

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 24ms
    fun minPartitions(n: String) =
        n.max() - '0'
```
```rust
// 0ms
    pub fn min_partitions(n: String) -> i32 {
        (n.bytes().max().unwrap() - 48) as _
    }
```

