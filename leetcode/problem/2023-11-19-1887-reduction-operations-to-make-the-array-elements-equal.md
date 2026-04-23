---
layout: leetcode-entry
title: "1887. Reduction Operations to Make the Array Elements Equal"
permalink: "/leetcode/problem/2023-11-19-1887-reduction-operations-to-make-the-array-elements-equal/"
leetcode_ui: true
entry_slug: "2023-11-19-1887-reduction-operations-to-make-the-array-elements-equal"
---

[1887. Reduction Operations to Make the Array Elements Equal](https://leetcode.com/problems/reduction-operations-to-make-the-array-elements-equal/description/) medium
[blog post](https://leetcode.com/problems/reduction-operations-to-make-the-array-elements-equal/solutions/4304937/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19112023-1887-reduction-operations?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/d8f6b0dc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/410

#### Problem TLDR

Number of operations to decrease all elements to the next smallest

#### Intuition

The algorithm pretty much in a problem definition, just implement it.

#### Approach

* iterate from the second position, to simplify the initial conditions

#### Complexity

- Time complexity:
$$O(nlog())$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun reductionOperations(nums: IntArray): Int {
      nums.sort()
      return (nums.size - 2 downTo 0).sumBy {
        if (nums[it] < nums[it + 1]) nums.size - 1 - it else 0
      }
    }

```

