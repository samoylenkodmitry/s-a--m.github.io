---
layout: leetcode-entry
title: "1658. Minimum Operations to Reduce X to Zero"
permalink: "/leetcode/problem/2023-09-20-1658-minimum-operations-to-reduce-x-to-zero/"
leetcode_ui: true
entry_slug: "2023-09-20-1658-minimum-operations-to-reduce-x-to-zero"
---

[1658. Minimum Operations to Reduce X to Zero](https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/description/) medium
[blog post](https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/solutions/4067002/kotlin-slide/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20092023-1658-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/00045a69.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/345

#### Problem TLDR

Min suffix-prefix to make an `x`

#### Intuition

We can reverse the problem: find the middle of the array to make an `arr_sum() - x`. Now, this problem can be solved using a sliding window technique.

#### Approach

For more robust sliding window:
* use safe array iteration for the right border
* use explicit `windowSize` variable
* check the result every time

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minOperations(nums: IntArray, x: Int): Int {
      val targetSum = nums.sum() - x
      var windowSize = 0
      var currSum = 0
      var res = Int.MAX_VALUE
      nums.onEachIndexed { i, n ->
        currSum += n
        windowSize++
        while (currSum > targetSum && windowSize > 0)
          currSum -= nums[i - (windowSize--) + 1]
        if (currSum == targetSum)
          res = minOf(res, nums.size - windowSize)
      }
      return res.takeIf { it < Int.MAX_VALUE } ?: -1
    }

```

