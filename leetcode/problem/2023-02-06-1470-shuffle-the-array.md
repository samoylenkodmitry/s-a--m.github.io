---
layout: leetcode-entry
title: "1470. Shuffle the Array"
permalink: "/leetcode/problem/2023-02-06-1470-shuffle-the-array/"
leetcode_ui: true
entry_slug: "2023-02-06-1470-shuffle-the-array"
---

[1470. Shuffle the Array](https://leetcode.com/problems/shuffle-the-array/description/) easy

[blog post](https://leetcode.com/problems/shuffle-the-array/solutions/3151995/kotlin-two-pointers-o-n-space/)

```kotlin
    fun shuffle(nums: IntArray, n: Int): IntArray {
        val arr = IntArray(nums.size)
        var left = 0
        var right = n
        var i = 0
        while (i < arr.lastIndex) {
            arr[i++] = nums[left++]
            arr[i++] = nums[right++]
        }
        return arr
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/110
#### Intuition
Just do what is asked.
#### Approach
For simplicity, use two pointers for the source, and one for the destination.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$

