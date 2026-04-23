---
layout: leetcode-entry
title: "905. Sort Array By Parity"
permalink: "/leetcode/problem/2023-09-28-905-sort-array-by-parity/"
leetcode_ui: true
entry_slug: "2023-09-28-905-sort-array-by-parity"
---

[905. Sort Array By Parity](https://leetcode.com/problems/sort-array-by-parity/description/) easy
[blog post](https://leetcode.com/problems/sort-array-by-parity/solutions/4098759/kotlin-3-one-liners/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28092023-905-sort-array-by-parity?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/66be2d11.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/353

#### Problem TLDR

Sort an array by even-odd

#### Intuition

There are built-in functions. However, in an interview manual partition is expected: maintain the sorted border `l` and adjust it after swapping.

#### Approach

Let's write them all.

#### Complexity
- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
	// 1
	fun sortArrayByParity(nums: IntArray) = nums.also {
      var l = 0
      for (r in nums.indices) if (nums[r] % 2 == 0)
        nums[r] = nums[l].also { nums[l++] = nums[r] }
    }

    // 2
    fun sortArrayByParity(nums: IntArray) =
      nums.partition { it % 2 == 0 }.toList().flatten()

    // 3
    fun sortArrayByParity(nums: IntArray) = nums.sortedBy { it % 2 }
```

