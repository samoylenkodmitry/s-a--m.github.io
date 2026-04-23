---
layout: leetcode-entry
title: "704. Binary Search"
permalink: "/leetcode/problem/2023-04-01-704-binary-search/"
leetcode_ui: true
entry_slug: "2023-04-01-704-binary-search"
---

[704. Binary Search](https://leetcode.com/problems/binary-search/description/) easy

[blog post](https://leetcode.com/problems/binary-search/solutions/3364415/kotlin-tricks/)

```kotlin

fun search(nums: IntArray, target: Int): Int {
    var lo = 0
    var hi = nums.lastIndex
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        if (nums[mid] == target) return mid
        if (nums[mid] < target) lo = mid + 1
        else hi = mid - 1
    }
    return -1
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/167
#### Intuition
Just write binary search.

#### Approach
For more robust code:
* use including ranges `lo..hi`
* check the last condition `lo == hi`
* always check the exit condition `== target`
* compute `mid` without the integer overflow
* always move the boundary `mid + ` or `mid - 1`
* check yourself where to move the boundary, imagine moving closer to the `target`
#### Complexity
- Time complexity:
$$O(log_2(n))$$
- Space complexity:
$$O(1)$$

