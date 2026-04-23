---
layout: leetcode-entry
title: "2540. Minimum Common Value"
permalink: "/leetcode/problem/2024-03-09-2540-minimum-common-value/"
leetcode_ui: true
entry_slug: "2024-03-09-2540-minimum-common-value"
---

[2540. Minimum Common Value](https://leetcode.com/problems/minimum-common-value/description/) easy
[blog post](https://leetcode.com/problems/minimum-common-value/solutions/4846251/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09032024-2540-minimum-common-value?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vZHLrXTNNpw)
![image.png](/assets/leetcode_daily_images/26394475.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/533

#### Problem TLDR

First common number in two sorted arrays #easy

#### Intuition

There is a short solution with `Set` and more optimal with two pointers: move the lowest one.

#### Approach

Let's implement both of them.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, or O(n) for `Set` solution

#### Code

```kotlin

  fun getCommon(nums1: IntArray, nums2: IntArray) = nums1
    .toSet().let { s -> nums2.firstOrNull { it in  s}} ?: -1

```
```rust

    pub fn get_common(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
      let (mut i, mut j) = (0, 0);
      while i < nums1.len() && j < nums2.len() {
        if nums1[i] == nums2[j] { return nums1[i] }
        else if nums1[i] < nums2[j] { i += 1 } else { j += 1 }
      }; -1
    }

```

