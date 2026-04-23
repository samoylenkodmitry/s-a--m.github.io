---
layout: leetcode-entry
title: "1464. Maximum Product of Two Elements in an Array"
permalink: "/leetcode/problem/2023-12-12-1464-maximum-product-of-two-elements-in-an-array/"
leetcode_ui: true
entry_slug: "2023-12-12-1464-maximum-product-of-two-elements-in-an-array"
---

[1464. Maximum Product of Two Elements in an Array](https://leetcode.com/problems/maximum-product-of-two-elements-in-an-array/description/) easy
[blog post](https://leetcode.com/problems/maximum-product-of-two-elements-in-an-array/solutions/4393721/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12122023-1464-maximum-product-of?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/nyXU1WVpcuo)
![image.png](/assets/leetcode_daily_images/39792771.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/436

#### Intuition

We can sort, we can search twice for indices, we can scan once with two variables.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun maxProduct(nums: IntArray): Int = with(nums.indices){
    maxBy { nums[it] }.let { i ->
    (nums[i] - 1) * (nums[filter { it != i }.maxBy { nums[it] }] - 1)
  }}

```

