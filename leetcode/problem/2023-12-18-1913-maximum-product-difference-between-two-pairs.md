---
layout: leetcode-entry
title: "1913. Maximum Product Difference Between Two Pairs"
permalink: "/leetcode/problem/2023-12-18-1913-maximum-product-difference-between-two-pairs/"
leetcode_ui: true
entry_slug: "2023-12-18-1913-maximum-product-difference-between-two-pairs"
---

[1913. Maximum Product Difference Between Two Pairs](https://leetcode.com/problems/maximum-product-difference-between-two-pairs/description/) easy
[blog post](https://leetcode.com/problems/maximum-product-difference-between-two-pairs/solutions/4419716/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18122023-1913-maximum-product-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/dqZXeeme8fE)
![image.png](/assets/leetcode_daily_images/b8ae372b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/443

#### Problem TLDR

max * second_max - min * second_min

#### Intuition

We can sort an array, or just find max and second max in a linear way.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun maxProductDifference(nums: IntArray): Int {
    var (a, b, c, d) = listOf(0, 0, Int.MAX_VALUE, Int.MAX_VALUE)
    for (x in nums) {
      if (x > a) b = a.also { a = x } else if (x > b) b = x
      if (x < d) c = d.also { d = x } else if (x < c) c = x
    }
    return a * b - c * d
  }

```

