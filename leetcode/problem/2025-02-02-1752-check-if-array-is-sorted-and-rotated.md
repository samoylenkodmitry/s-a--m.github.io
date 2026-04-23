---
layout: leetcode-entry
title: "1752. Check if Array Is Sorted and Rotated"
permalink: "/leetcode/problem/2025-02-02-1752-check-if-array-is-sorted-and-rotated/"
leetcode_ui: true
entry_slug: "2025-02-02-1752-check-if-array-is-sorted-and-rotated"
---

[1752. Check if Array Is Sorted and Rotated](https://leetcode.com/problems/check-if-array-is-sorted-and-rotated/description/) easy
[blog post](https://leetcode.com/problems/check-if-array-is-sorted-and-rotated/solutions/6361204/kotlin-rust-by-samoylenkodmitry-ik0h/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02022025-1752-check-if-array-is-sorted?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/JinUO8zVByY)
![1.webp](/assets/leetcode_daily_images/038baca9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/883

#### Problem TLDR

Is array sorted and rotated? #easy

#### Intuition

Count of violations must be less than 2.

#### Approach

* check '<' condition instead of '>=' to avoid the corner cases
* let's golf

#### Complexity

- Time complexity:
$$O(n)$$, O(n^2) for Kotlin's solution

- Space complexity:
$$O(1)$$, O(n^2) for Kotlin

#### Code

```kotlin

    fun check(n: IntArray) =
        n.sorted() in n.indices.map { n.drop(it) + n.take(it) }

```
```rust

    pub fn check(n: Vec<i32>) -> bool {
        (0..n.len()).filter(|&i| n[i] > n[(i + 1) % n.len()]).count() < 2
    }

```
```c++

    bool check(vector<int>& n) {
        int c = 0, m = size(n);
        for (int i = 0; i < m; c += n[i] > n[++i % m]);
        return c < 2;
    }

```

