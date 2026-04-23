---
layout: leetcode-entry
title: "881. Boats to Save People"
permalink: "/leetcode/problem/2023-04-03-881-boats-to-save-people/"
leetcode_ui: true
entry_slug: "2023-04-03-881-boats-to-save-people"
---

[881. Boats to Save People](https://leetcode.com/problems/boats-to-save-people/description/) medium

[blog post](https://leetcode.com/problems/boats-to-save-people/solutions/3373007/kotlin-two-pointers/)

```kotlin

fun numRescueBoats(people: IntArray, limit: Int): Int {
    people.sort()
    var count = 0
    var lo = 0
    var hi = people.lastIndex
    while (lo <= hi) {
        if (lo < hi && people[hi] + people[lo] <= limit) lo++
        hi--
        count++
    }
    return count
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/169
#### Intuition
The optimal strategy comes from an intuition: for each `people[hi]` of a maximum weight, we can or can not add the one man `people[lo]` of a minimum weight.
#### Approach
Sort an array and move two pointers `lo` and `hi`.
* Careful with the ending condition, `lo == hi`
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(1)$$

