---
layout: leetcode-entry
title: "2348. Number of Zero-Filled Subarrays"
permalink: "/leetcode/problem/2023-03-21-2348-number-of-zero-filled-subarrays/"
leetcode_ui: true
entry_slug: "2023-03-21-2348-number-of-zero-filled-subarrays"
---

[2348. Number of Zero-Filled Subarrays](https://leetcode.com/problems/number-of-zero-filled-subarrays/description/) medium

[blog post](https://leetcode.com/problems/number-of-zero-filled-subarrays/solutions/3323224/kotlin-count-of-subarrays/)

```kotlin

fun zeroFilledSubarray(nums: IntArray): Long {
    var currCount = 0L
    var sum = 0L
    nums.forEach {
        if (it == 0) currCount++ else currCount = 0L
        sum += currCount
    }
    return sum
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/155
#### Intuition
Consider the following sequence: `0`, `00`, `000`. Each time we are adding another element to the end of the previous. For `0` count of subarrays $$c_1 = 1$$, for `00` it is $$c_2 = c_1 + z_2$$, where $$z_2$$ is a number of zeros. So, the math equation is $$c_i = c_{i-1} + z_i$$, or $$c_n = \sum_{i=0}^{n}z_i $$

#### Approach
We can count subarray sums, then add them to the result, or we can just skip directly to adding to the result.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

