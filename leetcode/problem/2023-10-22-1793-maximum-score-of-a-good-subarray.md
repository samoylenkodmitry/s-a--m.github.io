---
layout: leetcode-entry
title: "1793. Maximum Score of a Good Subarray"
permalink: "/leetcode/problem/2023-10-22-1793-maximum-score-of-a-good-subarray/"
leetcode_ui: true
entry_slug: "2023-10-22-1793-maximum-score-of-a-good-subarray"
---

[1793. Maximum Score of a Good Subarray](https://leetcode.com/problems/maximum-score-of-a-good-subarray/description/) hard
[blog post](https://leetcode.com/problems/maximum-score-of-a-good-subarray/solutions/4194715/kotlin-must-include-nums-k/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23102023-1793-maximum-score-of-a?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/3afa1f41.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/378

#### Problem TLDR

Max of `window_min * (window_size)` for window having `nums[k]`

#### Intuition

This is *not* a problem where you need to find a minimum of a sliding window.

By description, we must always include `nums[k]`. Let's start from here and try to optimally add numbers to the left and to the right of it.

#### Approach

* in an interview, it is safer to write 3 separate loops: move both pointers, then move two others separately:

```kotlin
      while (i > 0 && j < nums.lastIndex) ...
      while (i > 0) ...
      while (j < nums.lastIndex) ...
```

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maximumScore(nums: IntArray, k: Int): Int {
      var i = k
      var j = k
      var min = nums[k]
      return generateSequence { when {
        i == 0 && j == nums.lastIndex -> null
        i > 0 && j < nums.lastIndex -> if (nums[i - 1] > nums[j + 1]) --i else ++j
        i > 0 -> --i else -> ++j
      } }.maxOfOrNull {
        min = min(min, nums[it])
        min * (j - i + 1)
      } ?: nums[k]
    }

```

