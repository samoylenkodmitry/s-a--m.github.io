---
layout: leetcode-entry
title: "1791. Find Center of Star Graph"
permalink: "/leetcode/problem/2024-06-27-1791-find-center-of-star-graph/"
leetcode_ui: true
entry_slug: "2024-06-27-1791-find-center-of-star-graph"
---

[1791. Find Center of Star Graph](https://leetcode.com/problems/find-center-of-star-graph/description/) easy
[blog post](https://leetcode.com/problems/find-center-of-star-graph/solutions/5375299/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27062024-1791-find-center-of-star?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/oBFqZCzcyd8)
![2024-06-27_06-48_1.webp](/assets/leetcode_daily_images/5aae0615.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/652

#### Problem TLDR

Center of a start graph #easy

#### Intuition

It's just a common node between two edges.

#### Approach

Can you make it shorter?

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun findCenter(e: Array<IntArray>) =
        e[0].first { it in e[1] }

```
```rust

    pub fn find_center(e: Vec<Vec<i32>>) -> i32 {
       if e[1].contains(&e[0][0]) { e[0][0] } else { e[0][1] }
    }

```

