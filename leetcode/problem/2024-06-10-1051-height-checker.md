---
layout: leetcode-entry
title: "1051. Height Checker"
permalink: "/leetcode/problem/2024-06-10-1051-height-checker/"
leetcode_ui: true
entry_slug: "2024-06-10-1051-height-checker"
---

[1051. Height Checker](https://leetcode.com/problems/height-checker/description/) easy
[blog post](https://leetcode.com/problems/height-checker/solutions/5287009/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10062024-1051-height-checker?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9HWfTXU-o4c)
![2024-06-10_06-29_1.webp](/assets/leetcode_daily_images/42842172.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/635

#### Problem TLDR

Count unsorted elements in array #easy

#### Intuition

We can use bucket sort to do this in O(n).

#### Approach

Let's just use a simple sort to save the effort.

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun heightChecker(heights: IntArray) = heights
        .toList().sorted().withIndex()
        .count { (i, h) -> h != heights[i] }

```
```rust

    pub fn height_checker(heights: Vec<i32>) -> i32 {
        let mut s = heights.clone(); s.sort_unstable();
        (0..s.len()).map(|i| (s[i] != heights[i]) as i32).sum()
    }

```

