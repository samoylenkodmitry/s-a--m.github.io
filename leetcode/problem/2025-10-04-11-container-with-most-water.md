---
layout: leetcode-entry
title: "11. Container With Most Water"
permalink: "/leetcode/problem/2025-10-04-11-container-with-most-water/"
leetcode_ui: true
entry_slug: "2025-10-04-11-container-with-most-water"
---

[11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/description) medium
[blog post](https://leetcode.com/problems/container-with-most-water/solutions/7247871/kotlin-rust-by-samoylenkodmitry-hei7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04102025-11-container-with-most-water?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7NLGgg6lg_U)

![1.webp](/assets/leetcode_daily_images/b7b9e67c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1132

#### Problem TLDR

Max water container from two heights #medium #two-pointers

#### Intuition

Start with two pointer at max distance. Decrease distance by 1 by moving the lower height pointer.
The length only decreases, so drop the lower height, it will not be better than the current.

#### Approach

* if heights are equal move any pointer or both (to change minimum)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 38ms
    fun maxArea(h: IntArray) =
        (h.lastIndex downTo 0).fold(0 to 0) { (r,i), l ->
            max(r,l*min(h[i],h[i+l])) to if (h[i]<h[i+l]) i+1 else i
        }.first

```
```rust

// 1ms
    pub fn max_area(h: Vec<i32>) -> i32 {
        (0..h.len()).rev().fold((0,0),|(r,i),l|
            (r.max(l as i32*h[i].min(h[i+l])),i+((h[i]<h[i+l])as usize))).0
    }

```

