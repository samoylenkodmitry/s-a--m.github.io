---
layout: leetcode-entry
title: "2439. Minimize Maximum of Array"
permalink: "/leetcode/problem/2023-04-05-2439-minimize-maximum-of-array/"
leetcode_ui: true
entry_slug: "2023-04-05-2439-minimize-maximum-of-array"
---

[2439. Minimize Maximum of Array](https://leetcode.com/problems/minimize-maximum-of-array/description/) medium

[blog post](https://leetcode.com/problems/minimize-maximum-of-array/solutions/3381720/kotlin-binary-search/)

```kotlin

fun minimizeArrayValue(nums: IntArray): Int {
    // 5 4 3 2 1 -> 5
    // 1 2 3 4 5 -> 3
    // 1 2 3 6 3
    // 1 2 6 3 3
    // 1 5 3 3 3
    // 3 3 3 3 3
    fun canArrangeTo(x: Long): Boolean {
        var diff = 0L
        for (i in nums.lastIndex downTo 0)
        diff = maxOf(0L, nums[i].toLong() - x + diff)
        return diff == 0L
    }
    var lo = 0
    var hi = Int.MAX_VALUE
    var min = Int.MAX_VALUE
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        if (canArrangeTo(mid.toLong())) {
            min = minOf(min, mid)
            hi = mid - 1
        } else lo = mid + 1
    }
    return min
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/171
#### Intuition
Observing the pattern, we can see, that any number from the `end` can be passed to the `start` of the array. One idea is to use two pointers, one pointing to the `biggest` value, another to the `smallest`. Given that biggest and smallest values changes, it will take $$O(nlog_2(n))$$ time to maintain such sorted structure.
Another idea, is that for any given `maximum value` we can walk an array from the end to the start and change values to be no bigger than it. This operation takes $$O(n)$$ time, and with the growth of the `maximum value` also increases a possibility to comply for all the elements. So, we can binary search in that space.
#### Approach
* careful with integers overflows
* for more robust binary search code:
* * check the final condition `lo == hi`
* * use inclusive `lo` and `hi`
* * always check the resulting value `min = minOf(min, mid)`
* * always move the borders `mid + 1` and `mid - 1`
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(1)$$

